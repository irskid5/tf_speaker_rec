import os
import json
from tensorboard.plugins.hparams import api as hp

# ----------------------------- General ------------------------------------
HP_SEED = hp.HParam('seed', hp.Discrete([1997]))  # for experimental reproducibility
RUNS_DIR = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/"
TB_LOGS_DIR = "logs/tensorboard/"
CKPT_DIR = "checkpoints/"
BACKUP_DIR = "tmp/backup"
RECORD_CKPTS = True
MULTI_STRATEGY = True

# ----------------------- Data preprocessing -------------------------------
# VoxCeleb1 specific
HP_SAMPLE_RATE = hp.HParam('sample_rate', hp.Discrete([16000.0]))
# VOXCELEB_DIR = "/media/vele/Data/Documents/University Files/Masters/Thesis/dev/datasets/VoxCeleb/VoxCeleb1"
VOXCELEB_DIR = "/data/dev/masters/data/VoxCeleb1"
HP_SHUFFLE_BUFFER_SIZE = hp.HParam('shuffle_buffer_size', hp.Discrete([32000])) # 134000 is total size

# Data selection
# Note: determines the maximum number of frames extracted from each utterance
#       for training
# HP_MAX_NUM_FRAMES = hp.HParam('max_num_frames', hp.Discrete([300, 400]))
HP_MAX_NUM_FRAMES = hp.HParam('max_num_frames', hp.Discrete([400]))

# Log-mel spectrograms
HP_NUM_MEL_BINS = hp.HParam('num_mel_bins', hp.Discrete([80]))
HP_FRAME_LENGTH = hp.HParam('frame_length', hp.Discrete([0.025]))  # in seconds
HP_FRAME_STEP = hp.HParam('frame_step', hp.Discrete([0.010]))  # in seconds
HP_UPPER_HERTZ = hp.HParam('upper_hertz', hp.Discrete([7600.0]))
HP_LOWER_HERTZ = hp.HParam('lower_hertz', hp.Discrete([80.0]))
HP_FFT_LENGTH = hp.HParam('fft_length', hp.Discrete([1024]))

# Input layer augmentation
HP_DOWNSAMPLE_FACTOR = hp.HParam('downsample_factor', hp.Discrete([3]))
HP_STACK_SIZE = hp.HParam('stack_size', hp.Discrete([4]))

# -------------------------------- Model -----------------------------------

# LSTM encoding layer
HP_NUM_LSTM_UNITS = hp.HParam('num_lstm_units', hp.Discrete([256*3])) # 256

# IRNN layer
HP_IRNN_STACK_SIZE = hp.HParam('irnn_stack_size', hp.Discrete([1]))
HP_IRNN_IDENTITY_SCALE = hp.HParam('irnn_identity_scale', hp.Discrete([0.1]))

# Attention aggregation layer
HP_NUM_SELF_ATT_UNITS = hp.HParam('num_self_att_units', hp.Discrete([512*3])) # 512
HP_NUM_SELF_ATT_HOPS = hp.HParam('num_self_att_hops', hp.Discrete([4]))

# Penultimate classification dense layers
HP_NUM_DENSE_UNITS = hp.HParam('num_dense_units', hp.Discrete([2048]))

# ------------------------------- Training ---------------------------------

# Training parameters
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64])) # , 64]))
# HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([2000]))

# Metrics
METRIC_TRAIN_LOSS = 'train_loss'
METRIC_TRAIN_ACCURACY = 'train_accuracy'
METRIC_EVAL_LOSS = 'eval_loss'
METRIC_EVAL_ACCURACY = 'eval_accuracy'
METRIC_EVAL_EER = 'eval_eer'
METRIC_ACCURACY = 'categorical_accuracy'
METRIC_TOP_K_ACCURACY = 'top_k_categorical_accuracy'
METRIC_LOSS = 'sigmoid_focal_crossentropy'

# Helper functions

def load_hparams(model_dir):
    with open(os.path.join(model_dir, 'hparams.json'), 'r') as f:
        return json.load(f)


def save_hparams(hparams, model_dir):
    with open(os.path.join(model_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f)
