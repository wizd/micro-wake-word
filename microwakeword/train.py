# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from absl import logging

import numpy as np
import tensorflow as tf

# NumPy 2 renamed trapz -> trapezoid; keep TF 2.17 (numpy<2) compatible.
_trapezoid = getattr(np, "trapezoid", np.trapz)

_EVAL_BATCH_SIZE = 4096
_PROGRESS_INTERVAL = 50


def _metric_numpy(metric):
    value = metric.result()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _metric_float(metric):
    return float(_metric_numpy(metric))


def _is_tensor(value):
    return tf.is_tensor(value)


def _consume_shuffle_rng(count):
    """Consume the same numpy shuffle stream get_data() would use."""
    if count <= 0:
        return
    indices = np.arange(count)
    np.random.shuffle(indices)


def _try_place_on_gpu(features, labels, name):
    nbytes = features.nbytes if hasattr(features, "nbytes") else 0
    try:
        with tf.device("/GPU:0"):
            gpu_features = tf.constant(features)
            gpu_labels = tf.constant(labels)
        # Touch the tensor so the copy is realized before we drop the numpy source.
        _ = gpu_features[:1]
        logging.info(
            "Cached %s on GPU: shape=%s (%.2f GiB)",
            name,
            tuple(gpu_features.shape),
            nbytes / (1024**3),
        )
        return gpu_features, gpu_labels
    except Exception as exc:
        logging.warning(
            "Keeping %s on host (%.2f GiB): %s",
            name,
            nbytes / (1024**3),
            exc,
        )
        return features, labels


def prepare_eval_cache(config, data_processor):
    """Load validation splits once (no extra numpy shuffle)."""
    val_x, val_y, _ = data_processor.get_data(
        "validation",
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
        shuffle=False,
    )
    val_y = val_y.reshape(-1, 1).astype(np.float32)
    val_x = np.asarray(val_x)
    cache = {
        "validation": _try_place_on_gpu(val_x, val_y, "validation"),
        "n_validation": int(val_y.shape[0]),
        "ambient": None,
        "n_ambient": 0,
    }

    if data_processor.get_mode_size("validation_ambient") > 0:
        ambient_x, ambient_y, _ = data_processor.get_data(
            "validation_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
            shuffle=False,
        )
        ambient_y = ambient_y.reshape(-1, 1).astype(np.float32)
        ambient_x = np.asarray(ambient_x)
        cache["ambient"] = _try_place_on_gpu(
            ambient_x, ambient_y, "validation_ambient"
        )
        cache["n_ambient"] = int(ambient_y.shape[0])

    return cache


def _evaluate_cached(eval_forward, metrics, features, labels, reset):
    """Forward a cached split and update `metrics` in place."""
    if reset:
        for metric in metrics:
            metric.reset_state()

    count = int(features.shape[0])
    for start in range(0, count, _EVAL_BATCH_SIZE):
        end = min(start + _EVAL_BATCH_SIZE, count)
        batch_x = features[start:end]
        batch_y = labels[start:end]
        if not _is_tensor(batch_x):
            batch_x = tf.constant(batch_x)
            batch_y = tf.constant(batch_y)
        predictions = eval_forward(batch_x)
        for metric in metrics:
            metric.update_state(batch_y, predictions)

    return {metric.name: _metric_numpy(metric) for metric in metrics}


def validate_nonstreaming(
    config,
    data_processor,
    model,
    test_set,
    eval_cache=None,
    eval_forward=None,
    metrics_list=None,
):
    if metrics_list is None:
        metrics_list = list(model.metrics)

    if eval_cache is None:
        testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
            test_set,
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="truncate_start",
        )
        testing_ground_truth = testing_ground_truth.reshape(-1, 1)
        val_features, val_labels = testing_fingerprints, testing_ground_truth
        n_validation = int(testing_ground_truth.shape[0])
        ambient_pair = None
        n_ambient = 0
        if data_processor.get_mode_size("validation_ambient") > 0:
            (
                ambient_testing_fingerprints,
                ambient_testing_ground_truth,
                _,
            ) = data_processor.get_data(
                test_set + "_ambient",
                batch_size=config["batch_size"],
                features_length=config["spectrogram_length"],
                truncation_strategy="split",
            )
            ambient_pair = (
                ambient_testing_fingerprints,
                ambient_testing_ground_truth.reshape(-1, 1),
            )
            n_ambient = int(ambient_pair[1].shape[0])
    else:
        val_features, val_labels = eval_cache["validation"]
        n_validation = eval_cache["n_validation"]
        ambient_pair = eval_cache["ambient"]
        n_ambient = eval_cache["n_ambient"]
        _consume_shuffle_rng(n_validation)
        if n_ambient:
            _consume_shuffle_rng(n_ambient)

    if eval_forward is None:
        @tf.function(jit_compile=True, reduce_retracing=True)
        def eval_forward(inputs):
            return model(inputs, training=False)

    result = _evaluate_cached(
        eval_forward, metrics_list, val_features, val_labels, reset=True
    )

    metrics = {}
    metrics["accuracy"] = result["accuracy"]
    metrics["recall"] = result["recall"]
    metrics["precision"] = result["precision"]

    metrics["auc"] = result["auc"]
    metrics["loss"] = result["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    test_set_fp = np.asarray(result["fp"])

    if n_ambient > 0 and ambient_pair is not None:
        ambient_features, ambient_labels = ambient_pair
        ambient_predictions = _evaluate_cached(
            eval_forward,
            metrics_list,
            ambient_features,
            ambient_labels,
            reset=False,
        )

        duration_of_ambient_set = (
            data_processor.get_mode_duration("validation_ambient") / 3600.0
        )

        # Other than the false positive rate, all other metrics are accumulated across
        # both test sets
        all_true_positives = np.asarray(ambient_predictions["tp"])
        ambient_false_positives = np.asarray(ambient_predictions["fp"]) - test_set_fp
        all_false_negatives = np.asarray(ambient_predictions["fn"])

        metrics["auc"] = ambient_predictions["auc"]
        metrics["loss"] = ambient_predictions["loss"]

        recall_at_cutoffs = (
            all_true_positives / (all_true_positives + all_false_negatives)
        )
        faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set

        target_faph_cutoff_probability = 1.0
        for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
            if faph_at_cutoffs[index] == 0:
                target_faph_cutoff_probability = cutoff
                recall_at_no_faph = recall_at_cutoffs[index]
                break

        if faph_at_cutoffs[0] > 2:
            # Use linear interpolation to estimate recall at 2 faph

            # Increase index until we find a faph less than 2
            index_of_first_viable = 1
            while faph_at_cutoffs[index_of_first_viable] > 2:
                index_of_first_viable += 1

            x0 = faph_at_cutoffs[index_of_first_viable - 1]
            y0 = recall_at_cutoffs[index_of_first_viable - 1]
            x1 = faph_at_cutoffs[index_of_first_viable]
            y1 = recall_at_cutoffs[index_of_first_viable]

            recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        else:
            # Lowest faph is already under 2, assume the recall is constant before this
            index_of_first_viable = 0
            recall_at_2faph = recall_at_cutoffs[0]

        x_coordinates = [2.0]
        y_coordinates = [recall_at_2faph]

        for index in range(index_of_first_viable, len(recall_at_cutoffs)):
            if faph_at_cutoffs[index] != x_coordinates[-1]:
                # Only add a point if it is a new faph
                # This ensures if a faph rate is repeated, we use the highest recall
                x_coordinates.append(faph_at_cutoffs[index])
                y_coordinates.append(recall_at_cutoffs[index])

        # Use trapezoid rule to estimate the area under the curve, then divide by 2.0 to get the average recall
        average_viable_recall = (
            _trapezoid(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
        )

        metrics["recall_at_no_faph"] = recall_at_no_faph
        metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
        metrics["ambient_false_positives"] = ambient_false_positives[50]
        metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
        metrics["average_viable_recall"] = average_viable_recall

    return metrics


def train(model, config, data_processor, prefetcher=None):
    # Assign default training settings if not set in the configuration yaml
    if not (training_steps_list := config.get("training_steps")):
        training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")):
        learning_rates_list = [0.001]
    if not (mix_up_prob_list := config.get("mix_up_augmentation_prob")):
        mix_up_prob_list = [0.0]
    if not (freq_mix_prob_list := config.get("freq_mix_augmentation_prob")):
        freq_mix_prob_list = [0.0]
    if not (time_mask_max_size_list := config.get("time_mask_max_size")):
        time_mask_max_size_list = [5]
    if not (time_mask_count_list := config.get("time_mask_count")):
        time_mask_count_list = [2]
    if not (freq_mask_max_size_list := config.get("freq_mask_max_size")):
        freq_mask_max_size_list = [5]
    if not (freq_mask_count_list := config.get("freq_mask_count")):
        freq_mask_count_list = [2]
    if not (positive_class_weight_list := config.get("positive_class_weight")):
        positive_class_weight_list = [1.0]
    if not (negative_class_weight_list := config.get("negative_class_weight")):
        negative_class_weight_list = [1.0]

    # Ensure all training setting lists are as long as the training step iterations
    def pad_list_with_last_entry(list_to_pad, desired_length):
        while len(list_to_pad) < desired_length:
            last_entry = list_to_pad[-1]
            list_to_pad.append(last_entry)

    training_step_iterations = len(training_steps_list)
    pad_list_with_last_entry(learning_rates_list, training_step_iterations)
    pad_list_with_last_entry(mix_up_prob_list, training_step_iterations)
    pad_list_with_last_entry(freq_mix_prob_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(positive_class_weight_list, training_step_iterations)
    pad_list_with_last_entry(negative_class_weight_list, training_step_iterations)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    cutoffs = np.linspace(0.0, 1.0, 101).tolist()

    accuracy_metric = tf.keras.metrics.BinaryAccuracy(name="accuracy")
    recall_metric = tf.keras.metrics.Recall(name="recall")
    precision_metric = tf.keras.metrics.Precision(name="precision")
    tp_metric = tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs)
    fp_metric = tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs)
    tn_metric = tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs)
    fn_metric = tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs)
    auc_metric = tf.keras.metrics.AUC(name="auc")
    bce_metric = tf.keras.metrics.BinaryCrossentropy(name="loss")

    eval_metrics = [
        accuracy_metric,
        recall_metric,
        precision_metric,
        tp_metric,
        fp_metric,
        tn_metric,
        fn_metric,
        auc_metric,
        bce_metric,
    ]
    # Threshold metrics are only consumed at validation time; keep the train
    # graph to the five values that are actually logged each interval.
    train_metrics = [
        accuracy_metric,
        recall_metric,
        precision_metric,
        auc_metric,
        bce_metric,
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def train_step(features, labels, sample_weights):
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss_value = loss(labels, predictions, sample_weight=sample_weights)
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for metric in train_metrics:
            metric.update_state(labels, predictions, sample_weight=sample_weights)
        return loss_value

    @tf.function(jit_compile=True, reduce_retracing=True)
    def eval_forward(inputs):
        return model(inputs, training=False)

    # Configure checkpointer and restore if available
    checkpoint_directory = os.path.join(config["train_dir"], "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Configure TensorBoard summaries
    train_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "train")
    )
    validation_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "validation")
    )

    logging.info("Preparing cached validation tensors")
    eval_cache = prepare_eval_cache(config, data_processor)

    training_steps_max = np.sum(training_steps_list)

    best_minimization_quantity = 10000
    best_maximization_quantity = 0.0
    best_no_faph_cutoff = 1.0

    use_prefetch = bool(prefetcher is not None and getattr(prefetcher, "started", False))

    for training_step in range(1, training_steps_max + 1):
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate = learning_rates_list[i]
                mix_up_prob = mix_up_prob_list[i]
                freq_mix_prob = freq_mix_prob_list[i]
                time_mask_max_size = time_mask_max_size_list[i]
                time_mask_count = time_mask_count_list[i]
                freq_mask_max_size = freq_mask_max_size_list[i]
                freq_mask_count = freq_mask_count_list[i]
                positive_class_weight = positive_class_weight_list[i]
                negative_class_weight = negative_class_weight_list[i]
                break

        optimizer.learning_rate.assign(learning_rate)

        augmentation_policy = {
            "mix_up_prob": mix_up_prob,
            "freq_mix_prob": freq_mix_prob,
            "time_mask_max_size": time_mask_max_size,
            "time_mask_count": time_mask_count,
            "freq_mask_max_size": freq_mask_max_size,
            "freq_mask_count": freq_mask_count,
        }

        if use_prefetch:
            (
                train_fingerprints,
                train_ground_truth,
                train_sample_weights,
            ) = prefetcher.get_batch(augmentation_policy)
        else:
            (
                train_fingerprints,
                train_ground_truth,
                train_sample_weights,
            ) = data_processor.get_data(
                "training",
                batch_size=config["batch_size"],
                features_length=config["spectrogram_length"],
                truncation_strategy="default",
                augmentation_policy=augmentation_policy,
            )

        train_ground_truth = np.asarray(train_ground_truth).reshape(-1)
        train_sample_weights = np.asarray(train_sample_weights).reshape(-1)
        combined_weights = (
            train_sample_weights
            * np.where(
                train_ground_truth,
                positive_class_weight,
                negative_class_weight,
            )
        ).reshape(-1, 1)
        train_ground_truth = train_ground_truth.reshape(-1, 1)

        train_step(
            tf.convert_to_tensor(train_fingerprints, dtype=tf.float32),
            tf.convert_to_tensor(train_ground_truth, dtype=tf.float32),
            tf.convert_to_tensor(combined_weights, dtype=tf.float32),
        )

        is_last_step = training_step == training_steps_max
        log_progress = (training_step % _PROGRESS_INTERVAL) == 0 or is_last_step
        if log_progress:
            # Host sync only on the progress cadence, not every step.
            print(
                "Validation Batch #{:d}: Accuracy = {:.3f}; Recall = {:.3f}; Precision = {:.3f}; Loss = {:.4f}; Mini-Batch #{:d}".format(
                    (training_step // config["eval_step_interval"] + 1),
                    _metric_float(accuracy_metric),
                    _metric_float(recall_metric),
                    _metric_float(precision_metric),
                    _metric_float(bce_metric),
                    (training_step % config["eval_step_interval"]),
                ),
                end="\r",
            )

        if (training_step % config["eval_step_interval"]) == 0 or is_last_step:
            step_accuracy = _metric_float(accuracy_metric)
            step_recall = _metric_float(recall_metric)
            step_precision = _metric_float(precision_metric)
            step_auc = _metric_float(auc_metric)
            step_loss = _metric_float(bce_metric)
            logging.info(
                "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                *(
                    training_step,
                    learning_rate,
                    step_accuracy * 100,
                    step_recall * 100,
                    step_precision * 100,
                    step_loss,
                ),
            )

            with train_writer.as_default():
                tf.summary.scalar("loss", step_loss, step=training_step)
                tf.summary.scalar("accuracy", step_accuracy, step=training_step)
                tf.summary.scalar("recall", step_recall, step=training_step)
                tf.summary.scalar("precision", step_precision, step=training_step)
                tf.summary.scalar("auc", step_auc, step=training_step)
                train_writer.flush()

            model.save_weights(
                os.path.join(config["train_dir"], "last_weights.weights.h5")
            )

            nonstreaming_metrics = validate_nonstreaming(
                config,
                data_processor,
                model,
                "validation",
                eval_cache=eval_cache,
                eval_forward=eval_forward,
                metrics_list=eval_metrics,
            )
            model.reset_metrics()  # reset metrics for next validation epoch of training
            logging.info(
                "Step %d (nonstreaming): Validation: recall at no faph = %.3f with cutoff %.2f, accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, ambient false positives = %d, estimated false positives per hour = %.5f, loss = %.5f, auc = %.5f, average viable recall = %.9f",
                *(
                    training_step,
                    nonstreaming_metrics["recall_at_no_faph"] * 100,
                    nonstreaming_metrics["cutoff_for_no_faph"],
                    nonstreaming_metrics["accuracy"] * 100,
                    nonstreaming_metrics["recall"] * 100,
                    nonstreaming_metrics["precision"] * 100,
                    nonstreaming_metrics["ambient_false_positives"],
                    nonstreaming_metrics["ambient_false_positives_per_hour"],
                    nonstreaming_metrics["loss"],
                    nonstreaming_metrics["auc"],
                    nonstreaming_metrics["average_viable_recall"],
                ),
            )

            with validation_writer.as_default():
                tf.summary.scalar(
                    "loss", nonstreaming_metrics["loss"], step=training_step
                )
                tf.summary.scalar(
                    "accuracy", nonstreaming_metrics["accuracy"], step=training_step
                )
                tf.summary.scalar(
                    "recall", nonstreaming_metrics["recall"], step=training_step
                )
                tf.summary.scalar(
                    "precision", nonstreaming_metrics["precision"], step=training_step
                )
                tf.summary.scalar(
                    "recall_at_no_faph",
                    nonstreaming_metrics["recall_at_no_faph"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "auc",
                    nonstreaming_metrics["auc"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "average_viable_recall",
                    nonstreaming_metrics["average_viable_recall"],
                    step=training_step,
                )
                validation_writer.flush()

            os.makedirs(os.path.join(config["train_dir"], "train"), exist_ok=True)

            model.save_weights(
                os.path.join(
                    config["train_dir"],
                    "train",
                    f"{int(best_minimization_quantity * 10000)}_weights_{training_step}.weights.h5",
                )
            )

            current_minimization_quantity = 0.0
            if config["minimization_metric"] is not None:
                current_minimization_quantity = nonstreaming_metrics[
                    config["minimization_metric"]
                ]
            current_maximization_quantity = nonstreaming_metrics[
                config["maximization_metric"]
            ]
            current_no_faph_cutoff = nonstreaming_metrics["cutoff_for_no_faph"]

            # Save model weights if this is a new best model
            if (
                (
                    (
                        current_minimization_quantity <= config["target_minimization"]
                    )  # achieved target false positive rate
                    and (
                        (
                            current_maximization_quantity > best_maximization_quantity
                        )  # either accuracy improved
                        or (
                            best_minimization_quantity > config["target_minimization"]
                        )  # or this is the first time we met the target
                    )
                )
                or (
                    (
                        current_minimization_quantity > config["target_minimization"]
                    )  # we haven't achieved our target
                    and (
                        current_minimization_quantity < best_minimization_quantity
                    )  # but we have decreased since the previous best
                )
                or (
                    (
                        current_minimization_quantity == best_minimization_quantity
                    )  # we tied a previous best
                    and (
                        current_maximization_quantity > best_maximization_quantity
                    )  # and we increased our accuracy
                )
            ):
                best_minimization_quantity = current_minimization_quantity
                best_maximization_quantity = current_maximization_quantity
                best_no_faph_cutoff = current_no_faph_cutoff

                # overwrite the best model weights
                model.save_weights(
                    os.path.join(config["train_dir"], "best_weights.weights.h5")
                )
                checkpoint.save(file_prefix=checkpoint_prefix)

            logging.info(
                "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
                best_minimization_quantity,
                (best_maximization_quantity * 100),
                best_no_faph_cutoff,
            )

    # Save checkpoint after training
    checkpoint.save(file_prefix=checkpoint_prefix)
    model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))
