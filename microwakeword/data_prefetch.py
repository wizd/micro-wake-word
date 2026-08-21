# coding=utf-8
"""Process-level training-batch prefetch.

Workers must be forked after FeatureHandler is loaded and before TensorFlow
initializes CUDA, so they inherit the mmap-backed datasets via copy-on-write
without pulling in a GPU context.
"""

from __future__ import annotations

import multiprocessing
import os
import random

from absl import logging

import numpy as np

_DEFAULT_POLICY = {
    "mix_up_prob": 0.0,
    "freq_mix_prob": 0.0,
    "time_mask_max_size": 0,
    "time_mask_count": 0,
    "freq_mask_max_size": 0,
    "freq_mask_count": 0,
}

_STOP = None


def _prefetch_worker(
    data_processor,
    request_q,
    result_q,
    batch_size,
    features_length,
    seed,
):
    random.seed(int(seed))
    np.random.seed(int(seed))
    while True:
        policy = request_q.get()
        if policy is _STOP:
            break
        try:
            batch = data_processor.get_data(
                "training",
                batch_size=batch_size,
                features_length=features_length,
                truncation_strategy="default",
                augmentation_policy=policy,
            )
            result_q.put(("ok", batch))
        except Exception as exc:  # pragma: no cover - surfaced to parent
            result_q.put(("err", repr(exc)))
            break


class TrainingPrefetcher:
    """Produce training batches in forked worker processes."""

    def __init__(
        self,
        data_processor,
        batch_size: int,
        features_length: int,
        num_workers: int | None = None,
        queue_size: int | None = None,
    ):
        self.data_processor = data_processor
        self.batch_size = int(batch_size)
        self.features_length = int(features_length)
        self.num_workers = int(
            num_workers
            if num_workers is not None
            else os.getenv("MICROWAKEWORD_PREFETCH_WORKERS", "3")
        )
        self.queue_size = int(
            queue_size
            if queue_size is not None
            else os.getenv("MICROWAKEWORD_PREFETCH_QUEUE", "6")
        )
        self._request_q = None
        self._result_q = None
        self._workers = []
        self._started = False

    @classmethod
    def from_config(cls, config: dict, data_processor):
        return cls(
            data_processor,
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
        )

    def start(self, initial_policy: dict | None = None) -> bool:
        if self._started:
            return True
        if self.num_workers <= 0:
            logging.info("Training prefetch disabled (num_workers=%d)", self.num_workers)
            return False

        try:
            start_method = multiprocessing.get_start_method(allow_none=True)
        except Exception:
            start_method = None
        if start_method is None:
            try:
                multiprocessing.set_start_method("fork")
                start_method = "fork"
            except RuntimeError:
                start_method = multiprocessing.get_start_method(allow_none=True)
        if start_method != "fork":
            logging.warning(
                "Training prefetch requires fork start method, got %s; "
                "falling back to in-process get_data",
                start_method,
            )
            return False

        ctx = multiprocessing.get_context("fork")
        self._request_q = ctx.Queue(maxsize=self.queue_size)
        self._result_q = ctx.Queue(maxsize=self.queue_size)
        base_seed = np.random.randint(1, 2**31 - 1)
        for i in range(self.num_workers):
            seed = (int(base_seed) + i * 100003) % (2**31 - 1)
            worker = ctx.Process(
                target=_prefetch_worker,
                args=(
                    self.data_processor,
                    self._request_q,
                    self._result_q,
                    self.batch_size,
                    self.features_length,
                    seed,
                ),
                name=f"mww-prefetch-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        policy = initial_policy or _DEFAULT_POLICY
        for _ in range(self.queue_size):
            self._request_q.put(policy)
        self._started = True
        logging.info(
            "Started %d training prefetch workers (queue=%d)",
            self.num_workers,
            self.queue_size,
        )
        return True

    @property
    def started(self) -> bool:
        return self._started

    def get_batch(self, augmentation_policy: dict):
        if not self._started:
            raise RuntimeError("TrainingPrefetcher.start() was not called")
        self._request_q.put(augmentation_policy)
        try:
            status, payload = self._result_q.get(timeout=120)
        except Exception as exc:
            alive = [w.is_alive() for w in self._workers]
            raise RuntimeError(
                f"prefetch queue timed out (workers alive={alive})"
            ) from exc
        if status != "ok":
            raise RuntimeError(f"prefetch worker failed: {payload}")
        return payload

    def stop(self) -> None:
        if not self._started:
            return
        for worker in self._workers:
            if worker.is_alive():
                worker.terminate()
        for worker in self._workers:
            worker.join(timeout=3)
        for queue in (self._request_q, self._result_q):
            if queue is None:
                continue
            try:
                queue.cancel_join_thread()
                queue.close()
            except Exception:
                pass
        self._workers = []
        self._started = False
        logging.info("Stopped training prefetch workers")
