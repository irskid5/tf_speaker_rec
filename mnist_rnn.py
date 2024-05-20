import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit("autoclustering")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from utils.model_utils import custom_sign_with_tanh_deriv_mod_on_inputs, sign_with_tanh_deriv
from mnist_rnn_model import get_model, get_ff_model

RUNS_DIR = "/home/vele/Documents/masters/mnist_rnn/runs/"
TB_LOGS_DIR = "logs/tensorboard/"
CKPT_DIR = "checkpoints/"
BACKUP_DIR = "tmp/backup"
RECORD_CKPTS = True

# Make sure we don't get any GPU errors
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
            print(str(e))

    # Set how many GPUs you want running
    # gpus = gpus[0:2]

    import random

    # Set training strategy (mirrored for multi, single for one, depending on availability
    # and multi_strategy flag)
    if multi_strategy and len(gpus) > 1:
        gpu_names = [x.name[len('/physical_device:'):] for x in gpus]
        print('Running multi gpu: {}'.format(', '.join(gpu_names)))
        strategy = tf.distribute.MirroredStrategy(
            devices=gpu_names)
        num_workers = len(gpus)
    else:
        # device = gpus[random.randint(0,3)].name[len('/physical_device:'):]
        device = gpus[1].name[len('/physical_device:'):]
        print('Running single gpu: {}'.format(device))
        strategy = tf.distribute.OneDeviceStrategy(
            device=device)
        num_workers = 1

    return strategy, dtype, num_workers

def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label

def resize(image, label):
    return tf.image.resize(image, [128,128]), label

def augment(image, label):
    # if tf.random.uniform((), minval=0, maxval=1) < 0.1:
    #     image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label

def run_one_train(options, layer_options, lr):
    strategy, _, num_workers = configure_environment(gpu_names=None, fp16_run=False, multi_strategy=False)

    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 512*num_workers

    # Setup for train dataset
    ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
    # ds_train = ds_train.map(resize, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
    ds_train = ds_train.prefetch(AUTOTUNE)

    # Setup for test Dataset
    ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
    # ds_test = ds_test.map(resize, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=True)
    ds_test = ds_test.prefetch(AUTOTUNE)

    now = datetime.now()
    RUN_DIR = RUNS_DIR + now.strftime("%Y%m") + "/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    # Specify checkpoint dir
    checkpoint_dir = None
    if checkpoint_dir:
        run_dir = checkpoint_dir[0:-12]

    # Specify location of pretrained weights
    pretrained_weights = None

    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-134607/checkpoints/" # Original, tanh, default init, Adam(1e-4), L2(1e-4), 99%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-140942/checkpoints/" # Original, sign_with_tanh, 20221208-134607 init, Adam(1e-5, clipnorm=1), 98%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-150638/checkpoints/" # tern input, sign_with_tanh, 20221208-140942 init, Adam(1e-5, clipnorm=1), 99%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-154453/checkpoints/" # tern input, sign_with_tanh, 20221208-150638 init, tern(0.1) for all, NoAccRegV2(lm=0,k=8) -> QRNN_0:0.63,QRNN_1:0.54,DENSE_0:0.59, Adam(1e-5, clipnorm=1), 93%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221221-232808/checkpoints/" # tern input, sign_with_tanh, 20221208-154453 init, tern(0.1) for all, NoAccRegV2(lm=5e-4,k=8) -> QRNN_0:1.0,QRNN_1:0.89,DENSE_0:0.79, Adam(CosineDecay(1e-4, 500, alpha=0.1), clipnorm=1), 85%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221221-232819/checkpoints/" # tern input, sign_with_tanh, 20221208-154453 init, tern(0.1) for all, NoAccRegV2(lm=1e-3,k=8) -> QRNN_0:1.0,QRNN_1:0.95,DENSE_0:0.87, Adam(CosineDecay(1e-4, 500, alpha=0.1), clipnorm=1), 73%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221221-193404/checkpoints/" # tern input, sign_with_tanh, 20221208-154453 init, tern(0.1) for all, NoAccRegV2(lm=1e-4,k=8) -> QRNN_0:0.99,QRNN_1:0.64,DENSE_0:0.64, Adam(CosineDecay(1e-4, 500, alpha=0.1), clipnorm=1), 92%

    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221216-184318/checkpoints/" # Original, sign_with_tanh, 20221208-134607 init, NoAccRegV2(lm=5e-6,k=4) -> QRNN_0:1.0,QRNN_1:0.99,DENSE_0:0.95, Adam(CosineDecay(1e-4, 100, alpha=0.1), clipnorm=1), 98%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221220-123439/checkpoints/" # tern input, sign_with_tanh, 20221216-184318 init, NoAccRegV2(lm=5e-6,k=4) -> QRNN_0:1.0,QRNN_1:0.99,DENSE_0:0.95, Adam(CosineDecay(1e-4, 100, alpha=0.1), clipnorm=1), 98%

    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221222-193832/checkpoints/" # og
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221222-195050/checkpoints/" # og + sign
    pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221222-202422/checkpoints/" # og + sign + tern
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221223-150643/checkpoints/" # og + sign + tern + quant, 92%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202302/20230205-160713/checkpoints/"
    pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202302/20230205-190604/checkpoints/"

    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221229-123124/checkpoints/" # og + sign + tern + quant + NoAccReg1(0.001 all, 0.005 Dense0), 20221223-150643 init, 71%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221229-144111/checkpoints/" # og + sign + tern + quant + NoAccReg2(0.001 all, 0.005 Dense0), 20221223-150643 init, 82%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221229-182840/checkpoints/" # og + sign + tern + (quant + NoAccReg2(0.001 all, 0.005 Dense0)), 20221222-202422 init, 84.48%

    # MODELS WITH ENLARGED INPUTS (128x128)
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202304/20230401-152516/checkpoints/" # og (128x128 resize), bs=512, cos(1e-4, 100, 1e-1), default rnn init, 98%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202304/20230401-181310/checkpoints/" # og (128x128 resize) + sign, bs=512, cos(5e-5, 100, 1e-1), 20230401-181310 init, 98%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230731-102905/checkpoints/" # og (128x128 resize) + sign + tern, bs=512, cos(5e-5, 100, 1e-1), 20230401-152516 init, 98%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202304/20230401-190335/checkpoints/" # og (128x128 resize) + sign + 5-bit sym input, bs=512, cos(5e-5, 100, 1e-1), 20230401-152516 init, 98%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202304/20230402-145048/checkpoints/" # og (128x128 resize) + sign + 5-bit sym input + quant + NoAccRegV2(lm=1e-4, k=6), bs=512, cos(5e-7, 100, 1e-1), 20230401-190335 init, 92%

    # MODELS WITH ENLARGED RNN LAYER (128->768)
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202304/20230403-105013/checkpoints/" # og, bs=512, cos(1e-4, 100, 1e-1), default rnn init, 99%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202304/20230403-162231/checkpoints/" # og + sign, bs=512, cos(1e-5, 100, 1e-1), 20230403-105013 init, 98%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202304/20230403-191924/checkpoints/" # og + sign + 5-bit sym input, bs=512, cos(1e-5, 100, 1e-1), 20230403-162231 init, 98%

    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202304/20230404-102137/checkpoints/" # og + sign + 5-bit sym input + quant, bs=512, cos(5e-7, 100, 1e-1), 20230403-191924 init, 92%

    # FF MODELS WITH 28x28->Dense(128)->Dense(128)->Dense(10)
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230724-221042/checkpoints/" # og, bs=512, default init, 96.9%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230724-221655/checkpoints/" # og + sign, 20230724-221042 init, 95.9%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230724-222239/checkpoints/" # og + sign + tern, 20230724-221655 init, 95.9%

    # FF MODELS WITH 28x28->Dense(512)->Dense(512)->Dense(10)
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230724-224845/checkpoints/" # og, bs=512, default init, 97.8%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230724-225511/checkpoints/" # og + sign, 20230724-224845 init, 97.5%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230724-230037/checkpoints/" # og + sign + tern, 20230724-225511 init, 97.4%

    # FF MODELS WITH 28x28->Dense(1024)->Dense(1024)->Dense(10)
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230725-142352/checkpoints/" # og, bs=512, default init, 98%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230725-150201/checkpoints/" # og + sign, 20230725-142352 init, 97.7%
    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202307/20230725-150421/checkpoints/" # og + sign + tern, 20230725-150201 init, 97.5%

    # pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202308/20230802-214429/checkpoints/"

    with strategy.scope():
        model = get_model(options, layer_options)
        # model = get_ff_model(options)

        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)
            print('Restored pretrained weights from {}.'.format(pretrained_weights))

        # Load weights if we starting from checkpoint
        if checkpoint_dir is not None:
            model.load_weights(checkpoint_dir)
            print('Restored checkpoint weights from {}.'.format(checkpoint_dir))

        # Reset the stat variables
        weights = model.get_weights()
        for i in range(len(weights)):
            if "/w" in model.weights[i].name or "/x" in model.weights[i].name:
                weights[i] = 0*weights[i]
        model.set_weights(weights)

        # Calculate total weight stats
        for layer in model.layers:
            if len(layer.trainable_weights) > 0:
                all_weights = tf.concat([tf.reshape(x, shape=[-1]) for x in layer.trainable_weights], axis=-1)
                tf.print("std   W  in ", layer.name, 1.0*tf.math.reduce_std(all_weights))
                tf.print("mean  W  in ", layer.name, 1.0*tf.math.reduce_mean(all_weights))
                tf.print("mean |W| in ", layer.name, 1.0*tf.math.reduce_mean(tf.abs(all_weights)))
                tf.print("")
        all_weights = tf.concat([tf.reshape(x, shape=[-1]) for x in model.trainable_weights], axis=-1)
        tf.print("std   W  all weights in ", model.name, 1.0*tf.math.reduce_std(all_weights))
        tf.print("mean  W  all weights in ", model.name, 1.0*tf.math.reduce_mean(all_weights))
        tf.print("mean |W| all weights in ", model.name, 1.0*tf.math.reduce_mean(tf.abs(all_weights)))

        # Print pruning percentage
        num_zeros = tf.constant(0, dtype=tf.int64)
        num_params = 0
        for i in range(len(weights)):
            if "/quantized" in model.weights[i].name:
                num_params += tf.size(model.weights[i])
                num_zeros += tf.cast(tf.size(model.weights[i]), dtype=tf.int64) - tf.cast(tf.math.count_nonzero(model.weights[i]), dtype=tf.int64)
                tf.print("Percentage of zeros: ", float(num_zeros)/float(num_params), " ", model.weights[i].name)
                num_params = 0
                num_zeros = 0
        # tf.print("\nNumber of parameters: ", num_params)
        # tf.print("Number of zeros: ", num_zeros)
        # tf.print("Percentage of zeros: ", float(num_zeros)/float(num_params))

        model.compile(
            # optimizer=larq.optimizers.Bop(threshold=1e-8, gamma=1e-4), # first ever binary optimizer (flips weights based on grads)
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    # model.run_eagerly = True

    # Define callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=(RUN_DIR + TB_LOGS_DIR), histogram_freq=1, update_freq="epoch", # profile_batch='100,200'
    )

    # Add a lr decay callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        tf.keras.optimizers.schedules.CosineDecay(5e-6, 100, alpha=0.1), #500
        verbose=0,
    )

    if RECORD_CKPTS:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(RUN_DIR + CKPT_DIR),
                                                    save_weights_only=True,
                                                    save_best_only=False,
                                                    monitor="val_accuracy",
                                                    mode="max",
                                                    verbose=1)
    else:
        ckpt_callback = None

    train = False
    test = True

    if train:
        try:
            model.fit(
                ds_train,
                epochs=2000,
                validation_data=ds_test,
                callbacks=[tb_callback, ckpt_callback, lr_callback],
                verbose=1,
            )
        except Exception as e:
            print(e)

    if test:
        try:
            for _ in range(10):
                model.evaluate(
                    x=ds_test,
                    verbose=1,
                )
        except:
            pass

tern_param_sets = {
    "mean_|W|_20221222_202422": {
        "QRNN_0": 0.0732264, 
        "QRNN_1": 0.0628442243, 
        "DENSE_0": 0.0307561122, 
        "DENSE_OUT": 0.0557919517,
    },
    "std_W_20221222_202422": {
        "QRNN_0": 0.10521546, 
        "QRNN_1": 0.0769046694, 
        "DENSE_0": 0.0381105281, 
        "DENSE_OUT": 0.0680549815,
    },
    "mean_|W|_20230401_190335": {
        "QRNN_0": 0.0467697, 
        "QRNN_1": 0.0462930948, 
        "DENSE_0": 0.00197623181, 
        "DENSE_OUT": 0.00746504217,
    },
    "mean_|W|_20230403_191924": {
        "QRNN_0": 0.00790670048, 
        "QRNN_1": 0.00594488857, 
        "DENSE_0": 0.00121824886, 
        "DENSE_OUT": 0.0463059247,
    },
    "mean_|W|_20230724_222239": {
        "DENSE_0": 0.0237941016, 
        "DENSE_1": 0.0606526956, 
        "DENSE_OUT": 0.166764185, 
    },
    "mean_|W|_20230724_230037": {
        "DENSE_0": 0.0296899974, 
        "DENSE_1": 0.0323777124, 
        "DENSE_OUT": 0.139907822, 
    },
    "mean_|W|_20230725_150421": {
        "DENSE_0": 0.00421457132, 
        "DENSE_1": 0.00742625585, 
        "DENSE_OUT": 0.0646042451, 
    },
    "mean_|W|_20230731_102905": {
        "QRNN_0": 0.0586392209, 
        "QRNN_1": 0.0515264273, 
        "DENSE_0": 0.00419227686, 
        "DENSE_OUT": 0.0201463439,
    },
    "mean_|W|_20230802_214429": {
        "QRNN_0": 0.0733767077,
        "QRNN_1": 0.0632859319,
        "DENSE_0": 0.0301486906,
        "DENSE_OUT": 0.0514439642,
    },
    "mean_|W|_20221222-193832": {
        "QRNN_0": 0.0766085088, 
        "QRNN_1": 0.0670150295, 
        "DENSE_0": 0.0286646392, 
        "DENSE_OUT": 0.048506543,
    },
    "default": {
        "QRNN_0": 0, 
        "QRNN_1": 0, 
        "SA_0_QDENSE_0": 0,
        "SA_0_QDENSE_1": 0,
        "DENSE_0": 0, 
        "DENSE_OUT": 0,
    },
}

# Optional
options = {
    "soft_thresh_tern": False,
    "learned_thresh": True,
    "tern": True,
    "tern_params": tern_param_sets["mean_|W|_20221222_202422"],
    "g": 1.5,
    "oar": {
        "use": True,
        "lm": 1e-4,
        "precision": 6,
    },
    "add_dist_loss": False,
}

# Activation functions
# options["activation_rnn"] = tf.keras.activations.tanh
# options["activation_dense"] = tf.keras.activations.tanh

# Activation fns (quantized)
# options["activation_rnn"] = sign_with_tanh_deriv
# options["activation_dense"] = sign_with_tanh_deriv

# Activation fns (quantized with mod)
options["activation_rnn"] = lambda x: custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=options["oar"]["precision"])
options["activation_dense"] = lambda x: custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=options["oar"]["precision"])

import threading
import time

# HARD CODED FOR THIS MODEL
layer_options = {
    "QRNN_0": {
        "activation": options["activation_rnn"],
        "add_dist_loss": options["add_dist_loss"],
        "oar": { 
            "use": options["oar"]["use"],
            "lm": options["oar"]["lm"],
            "precision": options["oar"]["precision"],
        },
        "s": 4.0,
        "tern_quant_thresh": options["g"]*options["tern_params"]["QRNN_0"], 
        "g": options["g"],
    }, 
    "QRNN_1": {
        "activation": options["activation_rnn"],
        "add_dist_loss": options["add_dist_loss"],
        "oar": { 
            "use": options["oar"]["use"],
            "lm": options["oar"]["lm"],
            "precision": options["oar"]["precision"],
        },
        "s": 4.0,
        "tern_quant_thresh": options["g"]*options["tern_params"]["QRNN_1"], 
        "g": options["g"],
    }, 
    
    # "SA_0_QDENSE_0": {
    #     "activation": options["activation_dense"],
    #     "add_dist_loss": options["add_dist_loss"],
    #     "oar": { 
    #         "use": options["oar"]["use"],
    #         "lm": options["oar"]["lm"],
    #         "precision": options["oar"]["precision"],
    #     },
    #     "s": 1,
    #     "tern_quant_thresh": options["g"]*options["tern_params"]["SA_0_QDENSE_0"], 
    #     "g": options["g"]
    # }, 
    # "SA_0_QDENSE_1": {
    #     "activation": options["activation_dense"],
    #     "add_dist_loss": options["add_dist_loss"],
    #     "oar": { 
    #         "use": options["oar"]["use"],
    #         "lm": options["oar"]["lm"],
    #         "precision": options["oar"]["precision"],
    #     },
    #     "s": 1,
    #     "tern_quant_thresh": options["g"]*options["tern_params"]["SA_0_QDENSE_1"], 
    #     "g": options["g"]
    # }, 

    "DENSE_0": {
        "activation": options["activation_dense"],
        "add_dist_loss": options["add_dist_loss"],
        "oar": { 
            "use": options["oar"]["use"],
            "lm": options["oar"]["lm"],
            "precision": options["oar"]["precision"],
        },
        "s": 1,
        "tern_quant_thresh": options["g"]*options["tern_params"]["DENSE_0"], 
        "g": options["g"]
    }, 
    "DENSE_OUT": {
        "activation": lambda x: tf.keras.activations.softmax(x),
        "add_dist_loss": False,
        "oar": { 
            "use": True,
            "lm": 0,
            "precision": options["oar"]["precision"],
        },
        "s": 1,
        "tern_quant_thresh": options["g"]*options["tern_params"]["DENSE_OUT"], 
        "g": options["g"],
    }
} 

run_one_train(options, layer_options, 5e-6)

# lrs = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024]
# oar_lms = [0, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]

# threads = []
# for lr in lrs:
#     for oar_lm in oar_lms:
#         oar_lm_real = 0.001 * oar_lm

#         options["oar"]["lm"] = oar_lm_real
#         layer_options["QRNN_0"]["oar"]["lm"] = oar_lm_real
#         layer_options["QRNN_1"]["oar"]["lm"] = oar_lm_real
#         layer_options["DENSE_0"]["oar"]["lm"] = oar_lm_real
#         layer_options["DENSE_OUT"]["oar"]["lm"] = oar_lm_real
        
#         print("LR = %f, OAR_LM = %f" % (lr, oar_lm))
#         thr = threading.Thread(target=run_one_train, args=(options, layer_options, 0.001*lr,))
#         threads.append(thr)
#         thr.start()

#         time.sleep(30)
#         # run_one_train(options, layer_options, 0.01*lr)
#     for index, thread in enumerate(threads):
#         print("Main    : before joining thread %d." % index)
#         thread.join()
#         print("Main    : thread %d done" % index)

print("End!")
