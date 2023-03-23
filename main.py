import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # info and warnings not printed

from absl import flags, logging, app
from tqdm import tqdm
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit("autoclustering")

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
import qkeras.callbacks as qcallbacks
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from pycm import *
from datetime import datetime
import larq

# DATA
import data
from preprocessing import get_dataset
# MODEL
from model import *
# QUANTIZATION
from quantization import *
import qkeras
# from utils.metrics import eer
# PARAMS
from config import *
# GENERAL UTILS
from utils.general import *

now = datetime.now()
RUN_DIR = RUNS_DIR + now.strftime("%Y%m") + "/" + now.strftime("%Y%m%d-%H%M%S") + "/"

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
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
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
    # gpus = gpus[0:2] # [0:2] is for 2 gpus

    # Log the placement of variables
    tf.debugging.set_log_device_placement(True)

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
            HP_IRNN_STACK_SIZE,
            HP_IRNN_IDENTITY_SCALE,
            HP_NUM_SELF_ATT_UNITS,
            HP_NUM_SELF_ATT_HOPS,
            HP_NUM_DENSE_UNITS,
            HP_SEED,
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
            HP_IRNN_STACK_SIZE: HP_IRNN_STACK_SIZE.domain.values[0],
            HP_IRNN_IDENTITY_SCALE: HP_IRNN_IDENTITY_SCALE.domain.values[0],
            HP_NUM_SELF_ATT_UNITS: HP_NUM_SELF_ATT_UNITS.domain.values[0],
            HP_NUM_SELF_ATT_HOPS: HP_NUM_SELF_ATT_HOPS.domain.values[0],
            HP_NUM_DENSE_UNITS: HP_NUM_DENSE_UNITS.domain.values[0],

            # Training
            HP_SEED: HP_SEED.domain.values[0],
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
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20220929-175346/checkpoints/" # Non-quantized best version
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20220930-161228/checkpoints/" # Input ternary (same acc as full)
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
    ds_train, ds_val, _, ds_info = get_dataset("voxceleb",
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
    num_classes = ds_info.features['label'].num_classes
    penultimate_layer_units = hparams[HP_NUM_DENSE_UNITS.name]
    with strategy.scope():
        # Get model
        model = get_model(hparams, num_classes-1, stateful=False, dtype=dtype)

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
            optimizer=tf.keras.optimizers.Adam(learning_rate=AttentionLRScheduler(2000, hparams[HP_NUM_LSTM_UNITS.name])),
            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            # loss=JointSoftmaxCenterLoss(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            loss={
                "DENSE_OUT": tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                # "DENSE_OUT": tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                "DENSE_0": CenterLoss(ratio=1, num_classes=(num_classes-1), num_features=penultimate_layer_units,from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            },
            loss_weights={"DENSE_OUT": 1, "DENSE_0": 0.0001},
            metrics={"DENSE_OUT": [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)]},
        )

        # model.run_eagerly = True

    # Define callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=(run_dir + TB_LOGS_DIR), histogram_freq=1, profile_batch='100,200'
    )

    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(run_dir + CKPT_DIR),
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    monitor="val_DENSE_OUT_categorical_accuracy",
                                                    mode="max",
                                                    verbose=1)
    else:
        ckpt_callback = None

    # NOTE: OneDeviceStrategy does not work with BackupAndRestore
    # fault_tol_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=(RUN_DIR + BACKUP_DIR))

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_DENSE_OUT_categorical_accuracy', min_delta=0.0001, patience=3, verbose=1
    )

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
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
                optimizer=tf.keras.optimizers.Adam(learning_rate=AttentionLRScheduler(2000, current_hparams[HP_NUM_LSTM_UNITS.name])),
                loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),  # True if last layer is not softmax
                metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)],
            )

        # Callbacks
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=run_dir, histogram_freq=1
        )

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(run_dir + CKPT_DIR),
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

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32)
    ]])
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

    loss_object = tf.keras.metrics.Mean()
    metric_objects = {fn.__name__: tf.keras.metrics.Mean() for fn in metrics}
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
    checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20221006-170430/checkpoints/"

    # Do we run eagerly for debugging? By default, we don't. Set "True" for here on.
    # tf.config.run_functions_eagerly(True)

    # Are we evaluating the full audio seq or not (meaning one segment of max_audio_size per file only)?
    eval_full = True

    # Init the environment
    # strategy, dtype, num_workers = configure_environment(
    #     gpu_names=None,
    #     fp16_run=False,
    #     multi_strategy=False)

    # Choose device (Stateful RNN only works non-distributed)
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    num_workers = 1
    dtype = tf.float32

    # Init hparams, choose to load from ckpt or config
    hparams, tb_hparams = setup_hparams(
        log_dir=checkpoint_dir+"../"+TB_LOGS_DIR,
        hparam_dir=checkpoint_dir+"../"+TB_LOGS_DIR)


    # Choose batch size (choose 1 if using eval_full)
    batch_size = 1 if eval_full else None
    if batch_size:
        hparams[HP_BATCH_SIZE.name] = batch_size

    # Load dataset !! CHOOSE DATASET HERE !!
    _, _, ds_test, ds_info = get_dataset("voxceleb",
                                        VOXCELEB_DIR,
                                        data.voxceleb,
                                        num_workers,
                                        strategy,
                                        hparams,
                                        is_hparam_search=False,
                                        eval_full=eval_full,
                                        dtype=dtype)

    lr = hparams[HP_LR.name]
    num_classes = ds_info.features['label'].num_classes
    penultimate_layer_units = hparams[HP_NUM_DENSE_UNITS.name]
    # with strategy.scope(): # STATEFUL RNN NOT SUPPORTED WITH DISTRIBUTE STRATEGY
    # Get model
    model = get_model(hparams, num_classes-1, stateful=eval_full, inference=True)

    # Load weights if we starting from checkpoint
    if checkpoint_dir is not None:
        model.load_weights(checkpoint_dir, by_name=False, skip_mismatch=False).expect_partial()
        logging.info('Restored weights from {}.'.format(checkpoint_dir))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=AttentionLRScheduler(2000, hparams[HP_NUM_LSTM_UNITS.name])),
        loss={
            "DENSE_OUT": tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            "DENSE_0": CenterLoss(ratio=1, num_classes=(num_classes-1), num_features=penultimate_layer_units,from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        },
        loss_weights={"DENSE_OUT": 1, "DENSE_0": 0.0001},
        metrics={"DENSE_OUT": [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)]},
    )
    # model.run_eagerly = True

    print("Start testing!")

    eval_dataset = ds_test
    eval_metrics = [tf.keras.metrics.categorical_accuracy, tf.keras.metrics.top_k_categorical_accuracy]
    with_cm = False
    eval_res = run_evaluate(
        model, 
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
        loss_fn=tf.keras.losses.categorical_crossentropy, 
        eval_dataset=eval_dataset.take(100), 
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

def quant_aware_training(_):
    # Get run dir
    run_dir = RUN_DIR

    # Initalize weights to a run for multi step training
    pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20220929-175346/checkpoints/"
    # pretrained_weights = None

    # Init the environment
    strategy, dtype, num_workers = configure_environment(
        gpu_names=None,
        fp16_run=False,
        multi_strategy=False)

    # Init hparams, choose to load from ckpt or config
    hparams, tb_hparams = setup_hparams(
        log_dir=(run_dir+TB_LOGS_DIR),
        hparam_dir=None)

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
    num_classes = ds_info.features['label'].num_classes
    penultimate_layer_units = hparams[HP_NUM_DENSE_UNITS.name]
    with strategy.scope():
        # Get model
        model = get_model(hparams, num_classes-1, stateful=False, dtype=dtype)

        # Load pretrained weights if necessary
        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            logging.info('Restored pretrained weights from {}.'.format(pretrained_weights))

        save_hparams(hparams, run_dir+TB_LOGS_DIR)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=AttentionLRScheduler(2000, hparams[HP_NUM_LSTM_UNITS.name])),
            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            # loss=JointSoftmaxCenterLoss(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            loss={
                "DENSE_OUT": tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                "DENSE_0": CenterLoss(ratio=1, num_classes=(num_classes-1), num_features=penultimate_layer_units,from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            },
            loss_weights={"DENSE_OUT": 1, "DENSE_0": 0.0001},
            metrics={"DENSE_OUT": [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)]},
        )

    # Retrieve the config
    config = model.get_config()

    with strategy.scope():
        # At loading time, register the custom objects with a `custom_object_scope`:
        custom_objects = {
            "IRNN": IRNN, 
            "TimeReduction": TimeReduction, 
            "BreakpointLayerForDebug": BreakpointLayerForDebug, 
            "sign_with_ste": sign_with_ste}
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.Model().from_config(config)

        with tfmot.quantization.keras.quantize_scope({
            "CustomQuantizeConfig": CustomQuantizeConfig,
            "IRNN": IRNN,
            "TimeReduction": TimeReduction,
            "BreakpointLayerForDebug": BreakpointLayerForDebug,
            "sign_with_ste": sign_with_ste,
        }):
        
            def apply_quantization_to_layers(layer):
                if isinstance(layer, tf.keras.layers.Lambda) or \
                    isinstance(layer, tf.keras.layers.Dot) or \
                    isinstance(layer, tf.keras.layers.Reshape) or \
                    isinstance(layer, TimeReduction):
                    return layer     
                return tfmot.quantization.keras.quantize_annotate_layer(layer, CustomQuantizeConfig())
            
            
            annotated_model = tf.keras.models.clone_model(
                model,
                clone_function=apply_quantization_to_layers,
            )

            quantized_model = tfmot.quantization.keras.quantize_apply(annotated_model)
            quantized_model.summary()

        quantized_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=AttentionLRScheduler(2000, hparams[HP_NUM_LSTM_UNITS.name])),
            loss={
                "quant_DENSE_OUT": tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                "quant_DENSE_0": CenterLoss(ratio=1, num_classes=(num_classes-1), num_features=penultimate_layer_units,from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            },
            loss_weights={"quant_DENSE_OUT": 1, "quant_DENSE_0": 0.0001},
            metrics={"quant_DENSE_OUT": [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)]},
        )

        # quantized_model.run_eagerly = True


    # Define callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=(run_dir + TB_LOGS_DIR), histogram_freq=1, # profile_batch='100,200'
    )

    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(run_dir + CKPT_DIR),
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    monitor="val_quant_DENSE_OUT_categorical_accuracy",
                                                    mode="max",
                                                    verbose=1)
    else:
        ckpt_callback = None

    # NOTE: OneDeviceStrategy does not work with BackupAndRestore
    # fault_tol_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=(RUN_DIR + BACKUP_DIR))

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_quant_DENSE_OUT_categorical_accuracy', min_delta=0.0001, patience=3, verbose=1
    )

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                           patience=4, min_lr=0.0001)

    # Train
    num_epochs = hparams[HP_NUM_EPOCHS.name]
    try:
        quantized_model.fit(
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

def post_quant_test(_, chk_dir=None):
    # Add checkpoint folder
    # checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20220930-161228/checkpoints/" # ternary input
    # checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20221004-125822/checkpoints/"
    # checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20221011-222803/checkpoints/" # ternary input, ternary recurrent kernel (thresh=0.33)
    # checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20221012-153610/checkpoints/" # reg input, htanh activation (72%)
    # checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20221019-170728/checkpoints/" # reg input, relu, SQ^2 w/ std (74% val)
    # checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/20221020-123140/checkpoints/" # ter input, relu, SQ^2 w/ std (70% val)
    # checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230113-110440/checkpoints/"

    # Override if value is passes to input
    checkpoint_dir = chk_dir if chk_dir else checkpoint_dir

    # Do we run eagerly for debugging? By default, we don't. Set "True" for here on.
    # tf.config.run_functions_eagerly(True)

    # Are we evaluating the full audio seq or not (meaning one segment of max_audio_size per file only)?
    eval_full = True

    # Init the environment
    strategy, dtype, num_workers = configure_environment(
        gpu_names=None,
        fp16_run=False,
        multi_strategy=False)

    # Choose device (Stateful RNN only works non-distributed)
    # strategy = tf.distribute.get_strategy()
    # num_workers = 1
    # dtype = tf.float32

    if not isinstance(checkpoint_dir, list):
        checkpoint_dir = [checkpoint_dir]

    # Init hparams, choose to load from ckpt or config
    hparams, tb_hparams = setup_hparams(
        log_dir=checkpoint_dir[0]+"../"+TB_LOGS_DIR,
        hparam_dir=checkpoint_dir[0]+"../"+TB_LOGS_DIR)


    # Choose batch size (choose 1 if using eval_full)
    batch_size = 1 if eval_full else None
    if batch_size:
        hparams[HP_BATCH_SIZE.name] = batch_size

    # Load dataset !! CHOOSE DATASET HERE !!
    _, _, ds_test, ds_info = get_dataset("voxceleb",
                                        VOXCELEB_DIR,
                                        data.voxceleb,
                                        num_workers,
                                        strategy,
                                        hparams,
                                        is_hparam_search=False,
                                        eval_full=eval_full,
                                        dtype=dtype)

    # g = [1.3, 1.4, 1.5, 2.0]

    # Load weights if we starting from checkpoint
    if checkpoint_dir is not None:
        for i, chkp in enumerate(checkpoint_dir):
            lr = hparams[HP_LR.name]
            num_classes = ds_info.features['label'].num_classes
            penultimate_layer_units = hparams[HP_NUM_DENSE_UNITS.name]
            # with strategy.scope(): # STATEFUL RNN NOT SUPPORTED WITH DISTRIBUTE STRATEGY
            # Get model
            model = get_model(hparams, num_classes-1, stateful=eval_full, inference=True)

            # model.load_weights(chkp, by_name=False, skip_mismatch=False).expect_partial()
            model.load_weights(chkp)
            tf.print('Restored weights from {}.'.format(chkp))

             # Reset the stat variables
            weights = model.get_weights()
            for i in range(len(weights)):
                if "/w" in model.weights[i].name or "/x" in model.weights[i].name or "/inp_" in model.weights[i].name or  "/out_" in model.weights[i].name:
                    weights[i] = 0*weights[i]
            model.set_weights(weights)

            # Quantize the weights (still floating point)
            """
            # print("Old Weights -----------------")
            # print_weight_stats(model)
            # model_weights = model.get_weights()
            # for i in range(len(model_weights)):
                # plot_histogram_continous(model_weights[i], "old_weights_dist.png")
                # old_mean = tf.reduce_mean(model_weights[i])
                # if "bias" in model.weights[i].name:
                    # continue
                # model_weights[i] = ternarize_tensor_with_threshold(model_weights[i], theta=2/3*tf.math.reduce_mean(tf.math.abs(model_weights[i])))
                # model_weights[i] = stochastic_binary()(model_weights[i])
                # new_mean = tf.reduce_mean(model_weights[i])
                # model_weights[i] = old_mean/new_mean*model_weights[i]
                # plot_histogram_discrete(model_weights[i], "new_weights_dist.png")
            # model.set_weights(model_weights)
            # print("New Weights -----------------")
            # print_weight_stats(model)
            """

            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=AttentionLRScheduler(2000, hparams[HP_NUM_LSTM_UNITS.name])),
                loss={
                    "DENSE_OUT": tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                    # "DENSE_0": CenterLoss(ratio=1, num_classes=(num_classes-1), num_features=penultimate_layer_units,from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
                },
                # loss_weights={"DENSE_OUT": 1, "DENSE_0": 0.0001},
                metrics={"DENSE_OUT": [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)]},
                jit_compile=True,
            )

            # model.run_eagerly = True
            
            print("Start testing!")

            eval_dataset = ds_test
            eval_metrics = [tf.keras.metrics.categorical_accuracy, tf.keras.metrics.top_k_categorical_accuracy]
            with_cm = False
            eval_res = run_evaluate(
                model, 
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                loss_fn=tf.keras.losses.categorical_crossentropy, 
                eval_dataset=eval_dataset, 
                batch_size=batch_size, 
                strategy=strategy,
                hparams=hparams,
                num_classes=num_classes-1,
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
            print('Final Results ({}): {}'.format(chkp, eval_res))
            print("End of Testing")

def qkeras_qat(_):
    # Get run dir
    run_dir = RUN_DIR

    # Set the checkpoint directory path if we are restarting from checkpoint
    checkpoint_dir = None
    if checkpoint_dir:
        run_dir = checkpoint_dir[0:-12]

    # This logs to tensorboard debugger (computationally heavy, data heavy),  seems to run in eager though
    # tf.debugging.experimental.enable_dump_debug_info(
    #     run_dir+TB_LOGS_DIR+"/tfdbg2_logdir",
    #     tensor_debug_mode="CONCISE_HEALTH",
    #     circular_buffer_size=-1)

    # No logging, throws error on NaN or inf in tensors, mildly comp. intense, seems to run in eager though
    # tf.debugging.enable_check_numerics(
    #     stack_height_limit=30, path_length_limit=50
    # )


    # Initalize weights to a run for multi step training
    pretrained_weights = None

    # WITH STACKED RNN 
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202209/20220929-175346/checkpoints/" # Original small param run
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202209/20220930-161228/checkpoints/" # Original small param run, ternary input
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221006-170430/checkpoints/" # 86% run, no quant, larger params
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221011-222803/checkpoints/" # ternary input, ternary recurrent kernel (thresh=0.33)
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221012-153610/checkpoints/" # reg input, htanh act, 82% full eval
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221013-223840/checkpoints/" # reg input, htanh act, RT2v2+RB2, 86% full eval
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221019-170728/checkpoints/" # reg input, relu, SQ^2 w/ std, 86% full eval
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221020-123140/checkpoints/" # ter input, relu, SQ^2 w/ std (70% val)
    
    # WITH JUST RNN
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221027-192721/checkpoints/" # Original
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221028-132315/checkpoints/" # Original w/ bn(rnn) 92% full eval
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221028-142648/checkpoints/" # ter input w/ bn(rnn)
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221028-122902/checkpoints/" # ter input
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221026-122933/checkpoints/" # Original w/ WN
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221026-152451/checkpoints/" # ter input w/ WN
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221031-153929/checkpoints/" # Original w/ bn(rnn), sign_htan act, 50%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202210/20221031-200447/checkpoints/" # Original w/ bn(all), SS(1), 74%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221103-234509/checkpoints/" # Original w/ SS(1), 77%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221118-161835/checkpoints/" # Original x/ sign_with_ss(1), 61%, clipnorm=1
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221118-231640/checkpoints/" # Original x/ sign_with_ss(1), 61%, clipnorm=1, ter_quant with +-x and 0*x
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221108-203218/checkpoints/" # ter input w/ SS(1), 74%, clipnorm=1
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221101-020512/checkpoints/" # Original  w/ bn(all), SS(3), 76%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221101-121631/checkpoints/" # Original  w/ bn(all), SS(5), 76%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221101-220703/checkpoints/" # Original  w/ bn(all), SS(8), 77%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221102-010642/checkpoints/" # Original w/ bn(all) after q, sign, tern_w_thresh weights, 46%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221104-111021/checkpoints/" # Original w/ SS(3), dist_loss, so far 38%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221104-130010/checkpoints/" # ter input w/ bn(all) after q, quant(all), SS(1), 51%, clipnorm=1
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221114-154622/checkpoints/" # ter input w/ bn(all) after q, quant(all), SS(1), 58%, clipnorm=1, 0.08 for rec + ker, 0.7*mean(abs) for rest
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221114-175306/checkpoints/" # ter input w/ bn(all) after q, quant(all), SS(1), 60%, clipnorm=1, 0.08 for rec + ker, 0.7*mean(abs) for rest, recorded deltas(norm)
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221114-181024/checkpoints/" # ter input w/ bn(all) after q, quant(all), sign with htan, 40%, clipnorm=1, 0.08 for rec + ker, 0.7*mean(abs) for rest, recorded deltas(norm)
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221114-230002/checkpoints/" # ter input w/ bn(all) after q, quant(all), sign with htan, 44%, clipnorm=1, 0.08 for rec + ker, 0.7*mean(abs) for rest, recorded deltas(norm)
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221113-142911/checkpoints/" # Original w/ ReLU, VarReg(2), 50%, clipnorm=1

    # LARGER NETWORKS
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221117-163745/checkpoints/" # Original (x4) w/ SS(1), clipnorm=0.1, 33%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221118-115744/checkpoints/" # ter input (x4) w/ SS(1), stochastic_ternary(t=0.25) quant, 2% so far but training
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221120-134420/checkpoints/" # Original (all x3 except dense, hops) w/ tanh, ATSCH(5400, HP_NUM_SELF_ATT_UNITS)), default rnn init, 75%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221125-184845/checkpoints/" # Original (all x3 except dense, hops) w/ tanh, lr=1e-4, dist_loss(10, kd=1, all else 0), default rnn init, 77%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221125-020548/checkpoints/" # Original (all x3 except dense, hops) w/ sign_with_tanh, lr=1e-4, clipnorm=1, dist_loss(10, kd=1, all else 0), 20221120-134420 init, 70%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221121-152133/checkpoints/" # Original (all x3 except dense, hops) w/ sign_with_htan, ATSCH(5400, HP_NUM_SELF_ATT_UNITS)), 20221120-134420 init, 65%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221126-153524/checkpoints/" # Original (all x3 except dense, hops) w/ sign_with_tanh, lr=1e-4, dist_loss(10, kd=1, all else 0), 20221125-184845 init, 67%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221125-112620/checkpoints/" # tern input (all x3 except dense, hops) w/ sign_with_tanh, lr=1e-4, dist_loss(10, kd=1, all else 0), 20221125-020548 init, 63%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221121-194336/checkpoints/" # tern input (all x3 except dense, hops) w/ sign_with_htan, ATSCH(5400, HP_NUM_SELF_ATT_UNITS)), 20221121-152133 init, 59%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221128-205042/checkpoints/" # tern input (all x3 except dense, hops) w/ sign_with_tanh, lr=1e-4, dist_loss(10, kd=1, all else 0), 20221126-153524 init, 60%

    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202212/20221202-124105/checkpoints/" # Original (all x3 except dense) w/ tanh, center_loss, lr=1e-4, dist_loss(0.01, kd=0, ks=0.01, kg=0), default rnn init, 63%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202212/20221202-130412/checkpoints/" # Original (all x3 except dense) w/ sign_with_tanh, lr=1e-5, dist_loss(0.01, kd=0, ks=0.01, kg=0), 20221202-124105 init, 62%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202212/20221202-145741/checkpoints/" # tern input (all x3 except dense) w/ sign_with_tanh, ATSCH(1000, 100000), dist_loss(0.01, kd=0, ks=0.01, kg=0), 20221202-130412 init, 61%

    # NETWORKS FOR SOFT THRESHOLD TERNARIZATION
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202212/20221203-213104/checkpoints/" # Original (// Ws (W1+W2)) w/ tanh, lr=1e-4, default rnn init, 65%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202212/20221207-162220/checkpoints/" # Original (// Ws (W1-W2)) w/ tanh, lr=1e-4, default rnn init, 60% but can still go

    # NETWORKS WITH ACTIVATION AFTER ATTENTION DOT PRODUCT
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202212/20221231-164111/checkpoints/" # Original (all x3 except dense, hops) w/ tanh, lr=1e-4, L2(0.0001), default rnn init, 79%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230102-131137/checkpoints/" # Original (all x3 except dense, hops) w/ sign_with_tanh, lr=1e-4, L2(0.0001), default rnn init, 63%    
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230103-154655/checkpoints/"
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230104-190554/checkpoints/" # Original (all x3 except dense, hops) w/ sign_with_tanh, act after dot, lr=1e-4, 20221125-020548 init, 65.78%    
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230105-030219/checkpoints/" # tern input (all x3 except dense, hops) w/ sign_with_tanh, act after dot, lr=1e-4, 20230104-190554 init, 59.81%   
    
    # NETWORKS FFROM LARGER NETWORKS WITH QUANT
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230113-134834/checkpoints/"
    # NOTE: The 44% fully-quantized run was 20221130-215651

    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230124-005909/checkpoints/" # bs=300, g=1.5 with different ones prop. to size of weights, NAR(1e-2, 11bit) only on Dense_0, lr=1e-7, clipnorm=1, sign_with_tanh_mod, 20221125-112620 init, 39%, 
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230124-005909/checkpoints/" # bs=300, g=1.5 with different ones prop. to size of weights, NAR(1e-3, 10bit) only on Dense_0, lr=0, clipnorm=1, sign_with_tanh_mod, 20230124-005909 init, 21%, 
    # pretrained_weights  = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230125-194623/checkpoints/" # bs=300, g=1.5 with different ones prop. to size of weights, NAR(1e-2, 8bit) on all exc. RNN0, lr=0, clipnorm=1, sign_with_tanh, 20230113-134834 init, bad %,

    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230127-100353/checkpoints/"
    
    # NETWORKS TRAINED WITH NAR ALL THE WAY THROUGH
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230127-225022/checkpoints/" # Original (all x3 except dense, hops) w/ sign_with_tanh, lr=cos(1e-4->1e-5, 60), NAR(8bit, 1e-3), bs=512, 20221120-134420 init, 66%
    pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230128-123500/checkpoints/" # tern input (all x3 except dense, hops) w/ sign_with_tanh, lr=cos(1e-4->1e-5, 60), NAR(8bit, 1e-3), bs=512, 20230127-225022 init, 62%

    # NETWORKS WITH DROPOUT
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202302/20230209-154352/checkpoints/" # Original (all x3 except dense, hops) w/ tanh, ATSCH(2000, 768), default rnn init, dropout=0.2, L2(1e-4), caching, 70%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202302/20230210-005047/checkpoints/" # Original (all x3 except dense, hops) w/ sign_with_tanh, cos(5e-5, 100, 1e-1), 20230209-154352 init, dropout=0, L2(1e-4), caching, 66%
    # pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202302/20230210-005047/checkpoints/" # tern input (all x3 except dense, hops) w/ sign_with_tanh, cos(5e-5, 100, 1e-1), 20230209-154352 init, dropout=0, L2(1e-4), caching, 62%

    # NETWORKS WITH ADD ATTENTION COMBO VS MULT
    pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202303/20230316-191739/checkpoints/" # Original (all x3 except dense, hops) w/ tanh, cos(1e-4, 100, 1e-1), default rnn init, L2(1e-4), caching, 68.51%
    pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202303/20230319-132807/checkpoints/" # Original (all x3 except dense, hops) w/ sign_with_tanh, cos(1e-4, 100, 1e-1), clipnorm=1, 20230316-191739 init, L2(1e-4), caching, 66.27%
    pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202303/20230320-021329/checkpoints/" # tern input (all x3 except dense, hops) w/ sign_with_tanh, cos(1e-4, 100, 1e-1), clipnorm=1, 20230319-132807 init, L2(1e-4), caching, 62%

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
    num_classes = ds_info.features['label'].num_classes
    penultimate_layer_units = hparams[HP_NUM_DENSE_UNITS.name]
    with strategy.scope():
        # Get model
        model = get_model(hparams, num_classes-1, stateful=False, dtype=dtype)

        # Load pretrained weights if necessary
        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            logging.info('Restored pretrained weights from {}.'.format(pretrained_weights))

        # Load weights if we starting from checkpoint
        if checkpoint_dir is not None:
            model.load_weights(checkpoint_dir)
            logging.info('Restored checkpoint weights from {}.'.format(checkpoint_dir))
            
        # Reset the stat variables
        # weights = model.get_weights()
        # for i in range(len(weights)):
        #     if "/w" in model.weights[i].name or "/x" in model.weights[i].name or "/inp_" in model.weights[i].name or "/out_" in model.weights[i].name:
        #         weights[i] = 0*weights[i]
        # model.set_weights(weights)

        # Calculate total weight stats
        # for layer in model.layers:
        #     if len(layer.trainable_weights) > 0:
        #         all_weights = tf.concat([tf.reshape(x, shape=[-1]) for x in layer.trainable_weights], axis=-1)
        #         tf.print("std   W  in ", layer.name, tf.math.reduce_std(all_weights))
        #         tf.print("std  |W| in ", layer.name, tf.math.reduce_std(tf.abs(all_weights)))
        #         tf.print("mean  W  in ", layer.name, tf.math.reduce_mean(all_weights))
        #         tf.print("mean |W| in ", layer.name, tf.math.reduce_mean(tf.abs(all_weights)))
        #         tf.print("")
        # all_weights = tf.concat([tf.reshape(x, shape=[-1]) for x in model.trainable_weights], axis=-1)
        # tf.print("std   W  all weights in ", model.name, tf.math.reduce_std(all_weights))
        # tf.print("std  |W| all weights in ", model.name, tf.math.reduce_std(tf.abs(all_weights)))
        # tf.print("mean  W  all weights in ", model.name, tf.math.reduce_mean(all_weights))
        # tf.print("mean |W| all weights in ", model.name, tf.math.reduce_mean(tf.abs(all_weights)))

        save_hparams(hparams, run_dir+TB_LOGS_DIR)

        model.compile(
            # optimizer=larq.optimizers.Bop(threshold=1e-3, gamma=1e-2), # first ever binary optimizer (flips weights based on grads)
            # optimizer=tf.keras.optimizers.Adam(learning_rate=AttentionLRScheduler(2000, 768)),
            optimizer=tf.keras.optimizers.Adam(clipnorm=1), # 1e-5 worked for quant with clipnorm=1
            loss={
                "DENSE_OUT": tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                # "DENSE_0": CenterLoss(ratio=1, num_classes=(num_classes-1), num_features=penultimate_layer_units, from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            },
            # loss_weights={"DENSE_OUT": 1, "DENSE_0": 0.0001},
            metrics={"DENSE_OUT": [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=5)]},
            jit_compile=True
        )

    # model.run_eagerly = True

    # Define callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=(run_dir + TB_LOGS_DIR), histogram_freq=1, update_freq="epoch", # profile_batch='10,15'
    )

    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        tf.keras.optimizers.schedules.CosineDecay(1e-4, 100, 1e-1), verbose=0
        # tf.keras.optimizers.schedules.CosineDecay(2e-6, 30, 1e-1), verbose=0 # for quant
        # tf.keras.optimizers.schedules.CosineDecay(5e-5, 100, 1e-1), verbose=0
    )

    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(run_dir + CKPT_DIR),
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    monitor="val_DENSE_OUT_categorical_accuracy",
                                                    mode="max",
                                                    verbose=1)
    else:
        ckpt_callback = None

    # NOTE: OneDeviceStrategy does not work with BackupAndRestore
    # fault_tol_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=(RUN_DIR + BACKUP_DIR))

    # early_stop_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_DENSE_OUT_categorical_accuracy', min_delta=0.0001, patience=3, verbose=1
    # )

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                           patience=4, min_lr=0.0001)

    # QKeras Noise Callback
    # qnoise_callback = qcallbacks.QNoiseScheduler(1, 200, freq_type="epoch", use_ste=True, log_dir=(run_dir + TB_LOGS_DIR))

    # Train
    train = True
    num_epochs = hparams[HP_NUM_EPOCHS.name]
    if train:
        try:
            model.fit(
                ds_train,
                epochs=num_epochs,
                validation_data=ds_val,
                callbacks=[tb_callback, ckpt_callback, lr_callback],
                verbose=1,
                # initial_epoch=77 # change for checkpoint !!!!
                # steps_per_epoch=20 # FOR TESTING TO MAKE EPOCH SHORTER
            )
        except Exception as e:
            print(e)
            print("In Exception")
            pass

    # Log results
    with tf.summary.create_file_writer((run_dir+TB_LOGS_DIR)).as_default():
        # Log hyperparameters
        hp.hparams(tb_hparams)

    # Add test at end of code
    test = False
    pretrained_weights = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230113-184346/checkpoints/"
    if test:
        post_quant_test(_, chk_dir=(pretrained_weights))
        # post_quant_test(_, chk_dir=(run_dir + CKPT_DIR))

def print_weight_stats(model):
    weights = model.weights
    for w in weights:
        print("Weight: ", w.name)
        print("Mean:   ", tf.math.reduce_mean(w).numpy())
        print("Std:    ", tf.math.reduce_std(w).numpy())
        print("Max:    ", tf.math.reduce_max(w).numpy())
        print("Min:    ", tf.math.reduce_min(w).numpy())
        print("")

if __name__ == '__main__':
    tf.config.run_functions_eagerly(False)
    tf.keras.backend.clear_session()
    # app.run(main)
    # app.run(hparam_search)
    # app.run(test_model)
    # app.run(quant_aware_training)
    # post_quant_test(
    #     None, 
    #     chk_dir=[
    #         "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202302/20230208-192733/checkpoints/"
    #         # "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202301/20230127-100353/checkpoints/", 
    #         # "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202211/20221125-112620/checkpoints/",
    #     ]
    # )
    app.run(qkeras_qat)
    print("End")

# CUSTOM TRAINING LOOP
"""
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_writer = tf.summary.create_file_writer(RUN_DIR+TB_LOGS_DIR+"train/")
val_writer = tf.summary.create_file_writer(RUN_DIR+TB_LOGS_DIR+"validation/")
train_step = val_step = 0

for epoch in range(num_epochs):
    # TRAINING -------------------------------------------------------------

    # Init progress bar
    print("\nepoch {}/{}".format(epoch + 1, num_epochs))
    prog_bar = tf.keras.utils.Progbar(ds_info.splits['train'].num_examples, stateful_metrics=metrics_names)

    # Training step
    for batch_idx, (x, y) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y, y_pred)

        # Update progress bar (per batch)
        values = [('train_loss', loss.numpy()), ('train_acc', acc_metric.result())]
        prog_bar.update(batch_idx * batch_size, values=values)

    # Freeze metrics
    train_loss = loss.numpy()
    train_acc = acc_metric.result().numpy()

    # Tensorboard logging (per epoch)
    with train_writer.as_default():
        tf.summary.scalar("Loss", train_loss, step=epoch+1)
        tf.summary.scalar(
            "Accuracy", train_acc, step=epoch+1,
        )
        train_step += 1

    # Reset accuracy in between epochs
    acc_metric.reset_states()

    # VALIDATION ------------------------------------------------------------

    # Iterate through validation set
    for batch_idx, (x, y) in enumerate(ds_val):
        y_pred = model(x, training=False)
        loss = loss_fn(y, y_pred)
        acc_metric.update_state(y, y_pred)

    # Freeze metrics
    val_loss = loss.numpy()
    val_acc = acc_metric.result().numpy()

    # Tensorboard logging (per epoch)
    with val_writer.as_default():
        tf.summary.scalar("Loss", val_loss, step=epoch+1)
        tf.summary.scalar(
            "Accuracy", val_acc, step=epoch+1,
        )
        val_step += 1

    # Update progress bar (per epoch)
    values = [('train_loss', train_loss), ('train_acc', train_acc), ('val_loss', val_loss), ('val_acc', val_acc)]
    prog_bar.update(ds_info.splits['train'].num_examples, values=values, finalize=True)

    # Reset accuracy final
    acc_metric.reset_states()
"""