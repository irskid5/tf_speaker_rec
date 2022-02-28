import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"

from absl import flags, logging, app
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# DATA
import data
from preprocessing import get_dataset
# MODEL
from model import *
# PARAMS
from config import *

RUN_DIR = RUNS_DIR + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# Init random seed for experimental reproducibility
tf.random.set_seed(HP_SEED.domain.values[0])
np.random.seed(HP_SEED.domain.values[0])


# Helper functions

def configure_environment(gpu_names, fp16_run=False, multi_strategy=False):
    # Set dtype
    dtype = tf.float32

    # ----------------- TO IMPLEMENT -------------------------
    # if fp16_run:
    #     print('Using 16-bit float precision.')
    #     policy = mixed_precision.Policy('mixed_float16')
    #     mixed_precision.set_policy(policy)
    #     dtype = tf.float16
    # --------------------------------------------------------

    # Get GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpu_names is not None and len(gpu_names) > 0:
        gpus = [x for x in gpus if x.name[len('/physical_device:'):] in gpu_names]

    # Init gpus
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            logging.warning(str(e))

    # Set training strategy (mirrored for multi, single for one, depending on availability
    # and multi_strategy flag)
    if multi_strategy and len(gpus) > 1:
        gpu_names = [x.name[len('/physical_device:'):] for x in gpus]
        print('Running multi gpu: {}'.format(', '.join(gpu_names)))
        strategy = tf.distribute.MirroredStrategy(
            devices=gpu_names)
        num_workers = len(gpus)
    else:
        device = gpus[1].name[len('/physical_device:'):]
        print('Running single gpu: {}'.format(device))
        strategy = tf.distribute.OneDeviceStrategy(
            device=device)
        num_workers = 1

    return strategy, dtype, num_workers


def setup_hparams(log_dir, checkpoint):
    if checkpoint is not None:
        checkpoint_dir = os.path.dirname(os.path.realpath(checkpoint))
        hparams = load_hparams(checkpoint_dir)

        tb_hparams = {}
        tb_keys = [
            HP_SAMPLE_RATE,
            HP_MAX_NUM_FRAMES,
            HP_NUM_MEL_BINS,
            HP_FRAME_LENGTH,
            HP_FRAME_STEP,
            HP_UPPER_HERTZ,
            HP_LOWER_HERTZ,
            HP_DOWNSAMPLE_FACTOR,
            HP_STACK_SIZE,
            HP_NUM_LSTM_UNITS,
            HP_NUM_DENSE_UNITS,
            HP_BATCH_SIZE,
            HP_LR,
            HP_NUM_EPOCHS,
            HP_SHUFFLE_BUFFER_SIZE
        ]

        for k, v in hparams.items():
            for tb_key in tb_keys:
                if k == tb_key.name:
                    tb_hparams[tb_key] = v

    else:
        tb_hparams = {
            # Preprocessing
            HP_SHUFFLE_BUFFER_SIZE: HP_SHUFFLE_BUFFER_SIZE.domain.values[0],
            HP_SAMPLE_RATE: HP_SAMPLE_RATE.domain.values[0],
            HP_MAX_NUM_FRAMES: HP_MAX_NUM_FRAMES.domain.values[0],
            HP_NUM_MEL_BINS: HP_NUM_MEL_BINS.domain.values[0],
            HP_FRAME_LENGTH: HP_FRAME_LENGTH.domain.values[0],
            HP_FRAME_STEP: HP_FRAME_STEP.domain.values[0],
            HP_UPPER_HERTZ: HP_UPPER_HERTZ.domain.values[0],
            HP_LOWER_HERTZ: HP_LOWER_HERTZ.domain.values[0],
            HP_FFT_LENGTH: HP_FFT_LENGTH.domain.values[0],
            HP_DOWNSAMPLE_FACTOR: HP_DOWNSAMPLE_FACTOR.domain.values[0],
            HP_STACK_SIZE: HP_STACK_SIZE.domain.values[0],

            # Model
            HP_NUM_LSTM_UNITS: HP_NUM_LSTM_UNITS.domain.values[0],
            HP_NUM_DENSE_UNITS: HP_NUM_DENSE_UNITS.domain.values[0],

            # Training
            HP_BATCH_SIZE: HP_BATCH_SIZE.domain.values[0],
            HP_LR: HP_LR.domain.values[0],
            HP_NUM_EPOCHS: HP_NUM_EPOCHS.domain.values[0]
        }

    with tf.summary.create_file_writer(os.path.join(log_dir, 'hparam_tuning')).as_default():
        hp.hparams_config(
            hparams=[
                HP_MAX_NUM_FRAMES,
                HP_NUM_LSTM_UNITS,
                HP_NUM_DENSE_UNITS,
                HP_LR,
                # HP_SHUFFLE_BUFFER_SIZE
            ],
            metrics=[
                hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                hp.Metric(METRIC_EER, display_name='EER'),
            ],
        )

    return {k.name: v for k, v in tb_hparams.items()}, tb_hparams


def main(_):
    # Set the checkpoint directory path if we are restarting from checkpoint
    checkpoint_dir = None

    # Init the environment
    strategy, dtype, num_workers = configure_environment(
        gpu_names=None,
        fp16_run=False,
        multi_strategy=MULTI_STRATEGY)

    # Init hparams, choose to load from ckpt or config
    hparams, tb_hparams = setup_hparams(
        log_dir=(RUN_DIR+TB_LOGS_DIR),
        checkpoint=checkpoint_dir)

    # Load dataset !! CHOOSE DATASET HERE !!
    ds_train, ds_val, ds_test, ds_info = get_dataset("voxceleb",
                                                     VOXCELEB_DIR,
                                                     data.voxceleb,
                                                     num_workers,
                                                     strategy,
                                                     hparams)
    # init training
    lr = hparams[HP_LR.name]
    with strategy.scope():
        # Get model
        model = get_model(hparams, ds_info.features['label'].num_classes)

        # Load weights if we starting from checkpoint
        if checkpoint_dir is not None:
            model.load_weights(checkpoint_dir)
            logging.info('Restored weights from {}.'.format(checkpoint_dir))

        save_hparams(hparams, RUN_DIR+TB_LOGS_DIR)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # True if last layer is not softmax
            metrics=["accuracy"],
        )

    # Log hyperparameters
    with tf.summary.create_file_writer((RUN_DIR+TB_LOGS_DIR)).as_default():
        hp.hparams(tb_hparams)

    # Define callbacks
    tb_callback = keras.callbacks.TensorBoard(
        log_dir=(RUN_DIR + TB_LOGS_DIR), histogram_freq=1, profile_batch='100,200'
    )

    if RECORD_CKPTS:
        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=(RUN_DIR + CKPT_DIR),
                                                    save_weights_only=True,
                                                    verbose=1)
    else:
        ckpt_callback = None

    # NOTE: OneDeviceStrategy does not work with BackupAndRestore
    # fault_tol_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=(RUN_DIR + BACKUP_DIR))

    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001, patience=4, verbose=1
    )

    # Train
    num_epochs = hparams[HP_NUM_EPOCHS.name]
    try:
        model.fit(
            ds_train,
            epochs=num_epochs,
            validation_data=ds_val,
            callbacks=[tb_callback, ckpt_callback, early_stop_callback],
            verbose=1,
            # steps_per_epoch=20 # FOR TESTING TO MAKE EPOCH SHORTER
        )
    except Exception as e:
        print(e)
        pass

def hparam_search(_):
    def run(run_dir, current_hparams):
        # Init the environment
        strategy, dtype, num_workers = configure_environment(
            gpu_names=None,
            fp16_run=False,
            multi_strategy=MULTI_STRATEGY)

        # Load dataset !! CHOOSE DATASET HERE !!
        ds_train, ds_val, ds_test, ds_info = get_dataset("voxceleb",
                                                         VOXCELEB_DIR,
                                                         data.voxceleb,
                                                         num_workers,
                                                         strategy,
                                                         current_hparams)

        # Init training
        with strategy.scope():
            # Get model
            model = get_model(current_hparams, ds_info.features['label'].num_classes)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=current_hparams[HP_LR.name]),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # True if last layer is not softmax
                metrics=["accuracy"],
            )

        # Callbacks
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=run_dir, histogram_freq=1
        )
        # hp_callback = hp.KerasCallback(run_dir, current_hparams)

        # Train
        num_epochs = 1
        try:
            model.fit(
                ds_train,
                epochs=num_epochs,
                callbacks=[tb_callback],
                verbose=1,
            )
        except Exception as e:
            print(e)
            pass

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            _, accuracy = model.evaluate(ds_val)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

        return None

    # Init hparams, choose to load from ckpt or config
    hparams, tb_hparams = setup_hparams(
        log_dir=(RUN_DIR + TB_LOGS_DIR),
        checkpoint=None)

    session_num = 0
    for max_num_frames in HP_MAX_NUM_FRAMES.domain.values:
        for num_lstm_units in HP_NUM_LSTM_UNITS.domain.values:
            for num_dense_units in HP_NUM_DENSE_UNITS.domain.values:
                for lr in HP_LR.domain.values:
                    # Set current hparams
                    hparams[HP_MAX_NUM_FRAMES.name] = max_num_frames
                    hparams[HP_NUM_LSTM_UNITS.name] = num_lstm_units
                    hparams[HP_NUM_DENSE_UNITS.name] = num_dense_units
                    hparams[HP_LR.name] = lr

                    # Log
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({HP_MAX_NUM_FRAMES.name: max_num_frames,
                           HP_NUM_LSTM_UNITS.name: num_lstm_units,
                           HP_NUM_DENSE_UNITS.name: num_dense_units,
                           HP_LR.name: lr})

                    # Run training
                    run(RUN_DIR + TB_LOGS_DIR + 'hparam_tuning/' + run_name, hparams)
                    session_num += 1


if __name__ == '__main__':
    # app.run(main)
    app.run(hparam_search)
    # print("End")

# CUSTOM TRAINING LOOP
# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# optimizer = keras.optimizers.Adam(learning_rate=lr)
# acc_metric = keras.metrics.SparseCategoricalAccuracy()
# train_writer = tf.summary.create_file_writer(RUN_DIR+TB_LOGS_DIR+"train/")
# val_writer = tf.summary.create_file_writer(RUN_DIR+TB_LOGS_DIR+"validation/")
# train_step = val_step = 0
#
# for epoch in range(num_epochs):
#     # TRAINING -------------------------------------------------------------
#
#     # Init progress bar
#     print("\nepoch {}/{}".format(epoch + 1, num_epochs))
#     prog_bar = keras.utils.Progbar(ds_info.splits['train'].num_examples, stateful_metrics=metrics_names)
#
#     # Training step
#     for batch_idx, (x, y) in enumerate(ds_train):
#         with tf.GradientTape() as tape:
#             y_pred = model(x, training=True)
#             loss = loss_fn(y, y_pred)
#
#         gradients = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#         acc_metric.update_state(y, y_pred)
#
#         # Update progress bar (per batch)
#         values = [('train_loss', loss.numpy()), ('train_acc', acc_metric.result())]
#         prog_bar.update(batch_idx * batch_size, values=values)
#
#     # Freeze metrics
#     train_loss = loss.numpy()
#     train_acc = acc_metric.result().numpy()
#
#     # Tensorboard logging (per epoch)
#     with train_writer.as_default():
#         tf.summary.scalar("Loss", train_loss, step=epoch+1)
#         tf.summary.scalar(
#             "Accuracy", train_acc, step=epoch+1,
#         )
#         train_step += 1
#
#     # Reset accuracy in between epochs
#     acc_metric.reset_states()
#
#     # VALIDATION ------------------------------------------------------------
#
#     # Iterate through validation set
#     for batch_idx, (x, y) in enumerate(ds_val):
#         y_pred = model(x, training=False)
#         loss = loss_fn(y, y_pred)
#         acc_metric.update_state(y, y_pred)
#
#     # Freeze metrics
#     val_loss = loss.numpy()
#     val_acc = acc_metric.result().numpy()
#
#     # Tensorboard logging (per epoch)
#     with val_writer.as_default():
#         tf.summary.scalar("Loss", val_loss, step=epoch+1)
#         tf.summary.scalar(
#             "Accuracy", val_acc, step=epoch+1,
#         )
#         val_step += 1
#
#     # Update progress bar (per epoch)
#     values = [('train_loss', train_loss), ('train_acc', train_acc), ('val_loss', val_loss), ('val_acc', val_acc)]
#     prog_bar.update(ds_info.splits['train'].num_examples, values=values, finalize=True)
#
#     # Reset accuracy final
#     acc_metric.reset_states()
