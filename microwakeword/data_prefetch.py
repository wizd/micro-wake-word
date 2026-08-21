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
from multiprocessing import shared_memory

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
    feature_shm_name,
    label_shm_name,
    weight_shm_name,
    feature_shape,
    label_shape,
    weight_shape,
    seed,
):
    feature_shm = shared_memory.SharedMemory(name=feature_shm_name)
    label_shm = shared_memory.SharedMemory(name=label_shm_name)
    weight_shm = shared_memory.SharedMemory(name=weight_shm_name)
    feature_slots = np.ndarray(feature_shape, dtype=np.float32, buffer=feature_shm.buf)
    label_slots = np.ndarray(label_shape, dtype=np.float64, buffer=label_shm.buf)
    weight_slots = np.ndarray(weight_shape, dtype=np.float64, buffer=weight_shm.buf)

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        while True:
            request = request_q.get()
            if request is _STOP:
                break
            step, slot, policy = request
            try:
                features, labels, weights = data_processor.get_data(
                    "training",
                    batch_size=feature_shape[1],
                    features_length=feature_shape[2],
                    truncation_strategy="default",
                    augmentation_policy=policy,
                )
                np.copyto(feature_slots[slot], features, casting="same_kind")
                np.copyto(label_slots[slot], labels, casting="unsafe")
                np.copyto(weight_slots[slot], weights, casting="unsafe")
                result_q.put(("ok", step, slot, None))
            except Exception as exc:  # pragma: no cover - surfaced to parent
                result_q.put(("err", step, slot, repr(exc)))
                break
    finally:
        feature_shm.close()
        label_shm.close()
        weight_shm.close()


class TrainingPrefetcher:
    """Produce training batches in forked worker processes."""

    def __init__(
        self,
        data_processor,
        batch_size: int,
        features_length: int,
        feature_bins: int,
        num_workers: int | None = None,
        queue_size: int | None = None,
    ):
        self.data_processor = data_processor
        self.batch_size = int(batch_size)
        self.features_length = int(features_length)
        self.feature_bins = int(feature_bins)
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
        self._free_slots = None
        self._workers = []
        self._shared_memory = []
        self._feature_slots = None
        self._label_slots = None
        self._weight_slots = None
        self._ready = {}
        self._submitted = set()
        self._started = False

    @classmethod
    def from_config(cls, config: dict, data_processor):
        return cls(
            data_processor,
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            feature_bins=config["training_input_shape"][-1],
        )

    def _allocate_shared_arrays(self) -> None:
        shapes_and_dtypes = (
            (
                (
                    self.queue_size,
                    self.batch_size,
                    self.features_length,
                    self.feature_bins,
                ),
                np.float32,
            ),
            ((self.queue_size, self.batch_size), np.float64),
            ((self.queue_size, self.batch_size), np.float64),
        )
        arrays = []
        for shape, dtype in shapes_and_dtypes:
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=size)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            self._shared_memory.append(shm)
            arrays.append(array)
        self._feature_slots, self._label_slots, self._weight_slots = arrays

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
        self._free_slots = ctx.Queue(maxsize=self.queue_size)
        self._allocate_shared_arrays()
        for slot in range(self.queue_size):
            self._free_slots.put(slot)

        feature_shape = self._feature_slots.shape
        label_shape = self._label_slots.shape
        weight_shape = self._weight_slots.shape
        base_seed = np.random.randint(1, 2**31 - 1)
        for i in range(self.num_workers):
            seed = (int(base_seed) + i * 100003) % (2**31 - 1)
            worker = ctx.Process(
                target=_prefetch_worker,
                args=(
                    self.data_processor,
                    self._request_q,
                    self._result_q,
                    self._shared_memory[0].name,
                    self._shared_memory[1].name,
                    self._shared_memory[2].name,
                    feature_shape,
                    label_shape,
                    weight_shape,
                    seed,
                ),
                name=f"mww-prefetch-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

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

    def submit(self, step: int, augmentation_policy: dict) -> None:
        if not self._started:
            raise RuntimeError("TrainingPrefetcher.start() was not called")
        if step in self._submitted:
            return
        slot = self._free_slots.get(timeout=120)
        self._request_q.put((int(step), int(slot), dict(augmentation_policy)))
        self._submitted.add(int(step))

    def get_batch(self, step: int):
        if not self._started:
            raise RuntimeError("TrainingPrefetcher.start() was not called")
        while step not in self._ready:
            try:
                status, result_step, slot, error = self._result_q.get(timeout=120)
            except Exception as exc:
                alive = [w.is_alive() for w in self._workers]
                raise RuntimeError(
                    f"prefetch queue timed out (workers alive={alive})"
                ) from exc
            if status != "ok":
                self._free_slots.put(slot)
                raise RuntimeError(f"prefetch worker failed at step {result_step}: {error}")
            self._ready[result_step] = slot

        slot = self._ready.pop(step)
        self._submitted.discard(step)
        return (
            self._feature_slots[slot],
            self._label_slots[slot],
            self._weight_slots[slot],
            slot,
        )

    def release(self, slot: int) -> None:
        if self._started:
            self._free_slots.put(int(slot))

    def stop(self) -> None:
        if not self._started:
            return
        for worker in self._workers:
            if worker.is_alive():
                worker.terminate()
        for worker in self._workers:
            worker.join(timeout=3)
        for queue in (self._request_q, self._result_q, self._free_slots):
            if queue is None:
                continue
            try:
                queue.cancel_join_thread()
                queue.close()
            except Exception:
                pass
        for shm in self._shared_memory:
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        self._workers = []
        self._shared_memory = []
        self._ready.clear()
        self._submitted.clear()
        self._started = False
        logging.info("Stopped training prefetch workers")
