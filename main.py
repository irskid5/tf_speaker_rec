from multiprocessing import reduction
import os
from pickle import TRUE
from tabnanny import check
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # info and warnings not printed

from absl import flags, logging, app
from tqdm import tqdm
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit(True)

import tensorflow_addons as tfa
from tensorflow import keras
from keras.utils import losses_utils
from tensorflow.keras import layers, models, optimizers, losses, metrics, mixed_precision, callbacks
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from pycm import *
from datetime import datetime

# DATA
import data
from preprocessing import get_dataset
# MODEL
from model import *
# from utils.metrics import eer
# PARAMS
from config import *
# GENERAL UTILS
from utils.general import *

RUN_DIR = RUNS_DIR + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# Init random seed for experimental reproducibility
tf.random.set_seed(HP_SEED.domain.values[0])
np.random.seed(HP_SEED.domain.values[0])


# Helper functions

def configure_environment(gpu_names, fp16_run=False, multi_strategy=False):
    # Set dtype
    dtype = tf.float32

    # ----------------- TO IMPLEMENT -------------------------
    if fp16_run:
        print('Using 16-bit float precision.')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        # dtype = tf.float16
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

    # Set how many GPUs you want running
    # gpus = gpus[0:2]

    # Set training strategy (mirrored for multi, single for one, depending on availability
    # and multi_strategy flag)
    if multi_strategy and len(gpus) > 1:
        gpu_names = [x.name[len('/physical_device:'):] for x in gpus]
        print('Running multi gpu: {}'.format(', '.join(gpu_names)))
        strategy = tf.distribute.MirroredStrategy(
            devices=gpu_names)
        num_workers = len(gpus)
    else:
        device = gpus[0].name[len('/physical_device:'):]
        print('Running single gpu: {}'.format(device))
        strategy = tf.distribute.OneDeviceStrategy(
            device=device)
        num_workers = 1

    return strategy, dtype, num_workers

def run_metrics(y_pred,
                y_true,
                metrics,
                strategy=None):

    return {
        metric_fn.__name__: metric_fn(y_true, y_pred)
        for metric_fn in metrics}


def setup_hparams(log_dir, hparam_dir):
    if hparam_dir is not None:
        hparams = load_hparams(hparam_dir)

        tb_hparams = {}
        tb_keys = [
            HP_SAMPLE_RATE,
            HP_MAX_NUM_FRAMES,
            HP_NUM_MEL_BINS,
            HP_FRAME_LENGTH,
            HP_FRAME_STEP,
            HP_UPPER_HERTZ,
            HP_LOWER_HERTZ,
            HP_FFT_LENGTH,
            HP_DOWNSAMPLE_FACTOR,
            HP_STACK_SIZE,
            HP_NUM_LSTM_UNITS,
            HP_NUM_SELF_ATT_UNITS,
            HP_NUM_SELF_ATT_HOPS,
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
            HP_NUM_SELF_ATT_UNITS: HP_NUM_SELF_ATT_UNITS.domain.values[0],
            HP_NUM_SELF_ATT_HOPS: HP_NUM_SELF_ATT_HOPS.domain.values[0],
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
                    HP_NUM_SELF_ATT_UNITS,
                    HP_NUM_SELF_ATT_HOPS,
                    HP_NUM_DENSE_UNITS,
                    # HP_LR,
                    # HP_SHUFFLE_BUFFER_SIZE
                ],
                metrics=[
                    hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                    hp.Metric(METRIC_TOP_K_ACCURACY, display_name='TopKAccuracy'),
                    hp.Metric(METRIC_LOSS, display_name='Loss'),
                    # hp.Metric(METRIC_EER, display_name='EER'),
                ],
            )

    return {k.name: v for k, v in tb_hparams.items()}, tb_hparams


def main(_):
    # Get run dir
    run_dir = RUN_DIR

    # Set the checkpoint directory path if we are restarting from checkpoint
    checkpoint_dir = None
    if checkpoint_dir:
        run_dir = checkpoint_dir[0:-12]

    # Initalize weights to a run for multi step training
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20220503-142802/checkpoints/"
    pretrained_weights = None

    # Init the environment
    strategy, dtype, num_workers = configure_environment(
        gpu_names=None,
        fp16_run=False,
        multi_strategy=MULTI_STRATEGY)

    # Init hparams, choose to load from ckpt or config
    hparams, tb_hparams = setup_hparams(
        log_dir=(run_dir+TB_LOGS_DIR),
        hparam_dir=checkpoint_dir + "../" + TB_LOGS_DIR if checkpoint_dir else None)

    # Load dataset !! CHOOSE DATASET HERE !!
    ds_train, ds_val, ds_test, ds_info = get_dataset("voxceleb",
                                                     VOXCELEB_DIR,
                                                     data.voxceleb,
                                                     num_workers,
                                                     strategy,
                                                     hparams,
                                                     is_hparam_search=False,
                                                     eval_full=False,
                                                     dtype=dtype)
    # init training
    lr = hparams[HP_LR.name]
    with strategy.scope():
        # Get model
        model = get_model(hparams, ds_info.features['label'].num_classes-1, stateful=False, dtype=dtype)

        # Load pretrained weights if necessary
        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            logging.info('Restored pretrained weights from {}.'.format(pretrained_weights))

        # Load weights if we starting from checkpoint
        if checkpoint_dir is not None:
            model.load_weights(checkpoint_dir)
            logging.info('Restored checkpoint weights from {}.'.format(checkpoint_dir))

        save_hparams(hparams, run_dir+TB_LOGS_DIR)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=AttentionLRScheduler(2000, hparams[HP_NUM_LSTM_UNITS.name])),
            # optimizer=optimizers.Adam(learning_rate=0.001),
            # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
            # loss=losses.CategoricalCrossentropy(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
            # loss=JointSoftmaxCenterLoss(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
            loss={
                "DENSE_OUT": losses.CategoricalCrossentropy(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
                # "DENSE_OUT": tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
                "DENSE_0": CenterLoss(ratio=1, from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
            },
            loss_weights={"DENSE_OUT": 1, "DENSE_0": 0.0001},
            metrics={"DENSE_OUT": [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy(k=5)]},
        )

        model.run_eagerly = True

    # Define callbacks
    tb_callback = callbacks.TensorBoard(
        log_dir=(run_dir + TB_LOGS_DIR), histogram_freq=1, profile_batch='100,200'
    )

    if RECORD_CKPTS:
        ckpt_callback = callbacks.ModelCheckpoint(filepath=(run_dir + CKPT_DIR),
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    monitor="val_DENSE_OUT_categorical_accuracy",
                                                    mode="max",
                                                    verbose=1)
    else:
        ckpt_callback = None

    # NOTE: OneDeviceStrategy does not work with BackupAndRestore
    # fault_tol_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=(RUN_DIR + BACKUP_DIR))

    early_stop_callback = callbacks.EarlyStopping(
        monitor='val_DENSE_OUT_categorical_accuracy', min_delta=0.0001, patience=3, verbose=1
    )

    # reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                           patience=4, min_lr=0.0001)

    # Train
    num_epochs = hparams[HP_NUM_EPOCHS.name]
    try:
        model.fit(
            ds_train,
            epochs=num_epochs,
            validation_data=ds_val,
            callbacks=[tb_callback, ckpt_callback],
            verbose=1,
            # initial_epoch=12 # change for checkpoint !!!!
            # steps_per_epoch=20 # FOR TESTING TO MAKE EPOCH SHORTER
        )
    except Exception as e:
        print(e)
        pass

    # Log results
    with tf.summary.create_file_writer((run_dir+TB_LOGS_DIR)).as_default():
        # Log hyperparameters
        hp.hparams(tb_hparams)


def hparam_search(_):
    def run(run_dir, current_hparams, strategy, ds_train, ds_val, ds_info):

        # Init training
        with strategy.scope():
            # Get model
            model = get_model(current_hparams, ds_info.features['label'].num_classes-1)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=AttentionLRScheduler(2000, current_hparams[HP_NUM_LSTM_UNITS.name])),
                loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),  # True if last layer is not softmax
                metrics=[metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy(k=5)],
            )

        # Callbacks
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=run_dir, histogram_freq=1
        )

        ckpt_callback = keras.callbacks.ModelCheckpoint(filepath=(run_dir + CKPT_DIR),
                                                    save_weights_only=True,
                                                    verbose=1)
        # hp_callback = hp.KerasCallback(run_dir, current_hparams)

        # Train
        num_epochs = 20 # !!Change this to test n epochs!!
        try:
            model.fit(
                ds_train,
                epochs=num_epochs,
                callbacks=[tb_callback, ckpt_callback],
                verbose=1,
            )
        except Exception as e:
            print(e)
            pass

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            loss, categorical_acc, top_k_categorical_acc = model.evaluate(ds_val)
            tf.summary.scalar(METRIC_LOSS, loss, step=1)
            tf.summary.scalar(METRIC_ACCURACY, categorical_acc, step=1)
            tf.summary.scalar(METRIC_TOP_K_ACCURACY, top_k_categorical_acc, step=1)

        return None

        # Init the environment
    strategy, dtype, num_workers = configure_environment(
        gpu_names=None,
        fp16_run=False,
        multi_strategy=MULTI_STRATEGY)

    # Init hparams, choose to load from ckpt or config
    hparams, tb_hparams = setup_hparams(
        log_dir=(RUN_DIR + TB_LOGS_DIR),
        hparam_dir=None)

    # !! Loop over hparams you want to test !!
    session_num = 0
    for max_num_frames in HP_MAX_NUM_FRAMES.domain.values:
        # Cache dataset
        hparams[HP_MAX_NUM_FRAMES.name] = max_num_frames
        ds_train, ds_val, _, ds_info = get_dataset("voxceleb",
                                                    VOXCELEB_DIR,
                                                    data.voxceleb,
                                                    num_workers,
                                                    strategy,
                                                    hparams,
                                                    is_hparam_search=True,
                                                    eval_full=False,
                                                    dtype=dtype)

        for num_lstm_units in HP_NUM_LSTM_UNITS.domain.values:
            for num_dense_units in HP_NUM_DENSE_UNITS.domain.values:
                # for lr in HP_LR.domain.values:
                # Set current hparams
                hparams[HP_NUM_LSTM_UNITS.name] = num_lstm_units
                hparams[HP_NUM_DENSE_UNITS.name] = num_dense_units
                # hparams[HP_LR.name] = lr

                # Log
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({HP_MAX_NUM_FRAMES.name: max_num_frames,
                        HP_NUM_LSTM_UNITS.name: num_lstm_units,
                        HP_NUM_DENSE_UNITS.name: num_dense_units})
                        # HP_LR.name: lr})

                # Run training
                run(RUN_DIR + TB_LOGS_DIR + 'hparam_tuning/' + run_name + "/", hparams, strategy, ds_train, ds_val, ds_info)
                session_num += 1


def run_evaluate(model,
                 optimizer,
                 loss_fn,
                 eval_dataset,
                 batch_size,
                 strategy,
                 hparams,
                 num_classes,
                 metrics=[],
                 fp16_run=False,
                 eval_full=False,
                 with_cm=False):

    feat_size = hparams[HP_NUM_MEL_BINS.name] * (hparams[HP_DOWNSAMPLE_FACTOR.name]+1)
        
    # @tf.function(input_signature=[[
    #     tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    #     tf.TensorSpec(shape=[None], dtype=tf.int64)
    # ]])
    def eval_step_framed(dist_inputs):
        def step_fn(inputs):
            x, y_true = inputs

            y_pred, _ = model(x, training=False)

            cm = None
            if with_cm:
                cm = tf.numpy_function(
                    calc_confusion_matrix, 
                    inp=[y_true, y_pred, num_classes, True], 
                    Tout=tf.int64)

            loss = loss_fn(y_true, y_pred)

            if metrics is not None:
                metric_results = run_metrics(y_pred=y_pred, y_true=y_true,
                    metrics=metrics, strategy=strategy)

            return loss, metric_results, cm

        loss, metrics_results, cm = strategy.run(step_fn, args=(dist_inputs,))
        loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, loss, axis=0)
        metrics_results = {name: strategy.reduce(
            tf.distribute.ReduceOp.MEAN, result, axis=0) for name, result in metrics_results.items()}

        return loss, metrics_results, cm

    # @tf.function(input_signature=[[
    #     tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    #     tf.TensorSpec(shape=[None], dtype=tf.int64)
    # ]])
    def eval_step_full(dist_inputs):
        def step_fn(inputs):
            x, y_true = inputs

            '''
            # For each segment, run prediction
            y_preds = None
            for i in range(tf.shape(x)[0]):
                y_pred = model(tf.reshape(x[i], [1, tf.shape(x[i])[0]]), training=False)
                # y_pred /= tf.reduce_max(y_pred)
                # y_pred_type = y_pred.dtype
                # y_pred = tf.where(
                #     # tf.equal(tf.reduce_max(y_pred, axis=1, keepdims=True), y_pred), 
                #     tf.greater_equal(y_pred, 0.9),
                #     tf.constant(1, shape=y_pred.shape), 
                #     tf.constant(0, shape=y_pred.shape)
                # )
                # y_pred = tf.cast(y_pred, dtype=y_pred_type)
                if y_preds == None:
                    y_preds = y_pred
                else:
                    y_preds = tf.concat([y_preds, y_pred], axis=0)
            
            y_pred_comb = tf.reduce_mean(y_preds, axis=0, keepdims=True)
            '''

            y_pred_comb, _ = model(x, training=False)

            model.reset_states()

            cm = None
            if with_cm:
                cm = tf.numpy_function(
                    calc_confusion_matrix, 
                    inp=[y_true, y_pred_comb, num_classes, True], 
                    Tout=tf.int64)

            loss = loss_fn(y_true, y_pred_comb)

            if metrics is not None:
                metric_results = run_metrics(
                    y_pred=y_pred_comb, 
                    y_true=tf.reshape(y_true, [1, -1]), 
                    metrics=metrics, strategy=strategy)

            return loss, metric_results, cm

        loss, metrics_results, cm = strategy.run(step_fn, args=(dist_inputs,))
        loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, loss, axis=0)
        metrics_results = {name: strategy.reduce(
            tf.distribute.ReduceOp.MEAN, result, axis=0) for name, result in metrics_results.items()}

        return loss, metrics_results, cm

    print('Performing evaluation.')

    loss_object = keras.metrics.Mean()
    metric_objects = {fn.__name__: keras.metrics.Mean() for fn in metrics}
    cm_full = np.zeros(shape=(num_classes, num_classes))

    for batch, inputs in enumerate(tqdm(eval_dataset)):

        if eval_full:
            loss, metrics_results, cm = eval_step_full(inputs)
        else: 
            loss, metrics_results, cm = eval_step_framed(inputs)

        loss_object(loss)
        for metric_name, metric_result in metrics_results.items():
            metric_objects[metric_name](metric_result)

        if with_cm:
            cm_full += cm

    final_results = {name: metric_object.result().numpy() for name, metric_object in metric_objects.items()}
    final_results[loss_fn.__name__] = loss_object.result().numpy()
    final_results["cm"] = cm_full.numpy() if with_cm else None

    return final_results


def test_model(_):
    # Add checkpoint folder
    checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20220503-142802/checkpoints/"

    # Do we run eagerly for debugging? By default, we don't. Set "True" for here on.
    # tf.config.run_functions_eagerly(True)

    # Init the environment
    # strategy, dtype, num_workers = configure_environment(
    #     gpu_names=None,
    #     fp16_run=False,
    #     multi_strategy=False)

    # Choose device
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    num_workers = 1
    dtype = tf.float32

    # Init hparams, choose to load from ckpt or config
    hparams, tb_hparams = setup_hparams(
        log_dir=checkpoint_dir+"../"+TB_LOGS_DIR,
        hparam_dir=checkpoint_dir+"../"+TB_LOGS_DIR)

    # Choose batch size (choose 1 if using eval_full)
    batch_size = 1
    if batch_size:
        hparams[HP_BATCH_SIZE.name] = batch_size

    # Are we evaluating the full audio seq or not (meaning one segment of max_audio_size per file only)?
    eval_full = True
    # Load dataset !! CHOOSE DATASET HERE !!
    ds_train, ds_val, ds_test, ds_info = get_dataset("voxceleb",
                                                     VOXCELEB_DIR,
                                                     data.voxceleb,
                                                     num_workers,
                                                     strategy,
                                                     hparams,
                                                     is_hparam_search=False,
                                                     eval_full=eval_full,
                                                     dtype=dtype)

    lr = hparams[HP_LR.name]
    # with strategy.scope(): # STATEFUL RNN NOT SUPPORTED WITH DISTRIBUTE STRATEGY
    # Get model
    model = get_model(hparams, ds_info.features['label'].num_classes-1, stateful=eval_full, inference=True)

    # Load weights if we starting from checkpoint
    if checkpoint_dir is not None:
        model.load_weights(checkpoint_dir, by_name=False, skip_mismatch=False).expect_partial()
        logging.info('Restored weights from {}.'.format(checkpoint_dir))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=AttentionLRScheduler(2000, hparams[HP_NUM_LSTM_UNITS.name])),
        # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
        # loss=losses.CategoricalCrossentropy(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
        # loss=JointSoftmaxCenterLoss(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
        loss={
            "DENSE_OUT": losses.CategoricalCrossentropy(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
            # "DENSE_OUT": tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE),
            "DENSE_0": CenterLoss(ratio=1, from_logits=True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
        },
        loss_weights={"DENSE_OUT": 1, "DENSE_0": 0.0001},
        metrics={"DENSE_OUT": [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy(k=5)]},
        # run_eagerly=True
    )
    # model.run_eagerly = True

    print("Start testing!")

    eval_dataset = ds_test
    eval_metrics = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]
    with_cm = False
    eval_res = run_evaluate(
        model, 
        optimizer=optimizers.Adam(learning_rate=lr), 
        loss_fn=losses.categorical_crossentropy, 
        eval_dataset=eval_dataset, 
        batch_size=batch_size, 
        strategy=strategy,
        hparams=hparams,
        num_classes=ds_info.features['label'].num_classes-1,
        metrics=eval_metrics, 
        fp16_run=False,
        eval_full=eval_full,
        with_cm=with_cm)

    # Confusion matrix analysis
    if with_cm:
        print("Confusion Matrix selected")
        print("Saving Confusion Matrix")
        run_date = checkpoint_dir.split('/')[1]
        cm_dict = {idx: {i: int(v) for i, v in enumerate(val)} for idx, val in enumerate(eval_res['cm'])}
        cm = ConfusionMatrix(matrix=cm_dict)
        print("Saving CSV")
        cm.save_csv("cm_"+run_date)
        print("Saving Stat")
        cm.save_stat("cm_stats_"+run_date)
        eval_res.pop('cm', None)

    # Show final results
    print("Final Results: ")
    print(eval_res)

    print("End of Testing")


if __name__ == '__main__':
    # tf.config.run_functions_eagerly(False)
    app.run(main)
    # app.run(hparam_search)
    # app.run(test_model)
    print("End")

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
