from tensorboard.plugins.hparams import api as hp

# ----------------------------- General ------------------------------------
HP_SEED = hp.HParam('seed', hp.Discrete([1997])) # for experimental reproducibility

# ----------------------- Data preprocessing -------------------------------
# VoxCeleb1 specific
HP_SAMPLE_RATE = hp.HParam('sample_rate', hp.Discrete([16000.0]))
VOXCELEB_DIR = "/media/vele/Data/Documents/University Files/Masters/Thesis/dev/datasets/VoxCeleb/VoxCeleb1"

# Data selection
# Note: determines the maximum number of frames extracted from each utterance
#       for training
HP_MAX_NUM_FRAMES = hp.HParam('max_num_frames', hp.Discrete([202]))

# Log-mel spectrograms
HP_NUM_MEL_BINS = hp.HParam('num_mel_bins', hp.Discrete([80]))
HP_FRAME_LENGTH = hp.HParam('frame_length', hp.Discrete([0.025])) # in seconds
HP_FRAME_STEP = hp.HParam('frame_step', hp.Discrete([0.010])) # in seconds
HP_UPPER_HERTZ = hp.HParam('upper_hertz', hp.Discrete([7600.0]))
HP_LOWER_HERTZ = hp.HParam('lower_hertz', hp.Discrete([80.0]))
HP_FFT_LENGTH = hp.HParam('fft_length', hp.Discrete([1024]))

# Input layer augmentation
HP_DOWNSAMPLE_FACTOR = hp.HParam('downsample_factor', hp.Discrete([3]))
HP_STACK_SIZE = hp.HParam('stack_size', hp.Discrete([4]))

# ------------------------------- Training ---------------------------------

# Training parameters
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-2,1e-3,1e-4,1e-5]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([1]))

# Metrics
METRIC_TRAIN_LOSS = 'train_loss'
METRIC_TRAIN_ACCURACY = 'train_accuracy'
METRIC_EVAL_LOSS = 'eval_loss'
METRIC_EVAL_ACCURACY = 'eval_accuracy'
METRIC_EVAL_EER = 'eval_eer'
METRIC_ACCURACY = 'accuracy'
METRIC_EER = 'eer'