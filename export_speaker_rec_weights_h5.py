import os

import tensorflow as tf
from tensorflow import keras

# DATA
import data
from preprocessing import import_dataset
# MODEL
from model import get_model
# HELPER FUNCTIONS
from main import setup_hparams
# PARAMS
from config import *

checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202305/20230518-161756/checkpoints/"

# Init hparams
hparams, tb_hparams = setup_hparams(
    log_dir=checkpoint_dir+"../"+TB_LOGS_DIR,
    hparam_dir=checkpoint_dir+"../"+TB_LOGS_DIR
)

# Load VoxCeleb
_, _, _, ds_info = import_dataset("voxceleb", VOXCELEB_DIR)
print("Loaded VoxCeleb info.")

num_classes = ds_info.features['label'].num_classes
model = get_model(hparams=hparams, num_classes=num_classes-1)
print("Loaded SpeakerRec model.")

if checkpoint_dir is not None:
    model.load_weights(checkpoint_dir)
    print('Restored pretrained weights from {}.'.format(checkpoint_dir))

# Reset the stat variables
weights = model.get_weights()
weights_to_discard = []
for i in range(len(weights)):
    if "/w" in model.weights[i].name or "/x" in model.weights[i].name:
        weights[i] = 0*weights[i]
model.set_weights(weights)

h5_dir = checkpoint_dir + "hdf5/"
if not os.path.exists(h5_dir):
    os.makedirs(h5_dir)
h5_filepath = h5_dir + "weights.hdf5"

print("Saving weights from -> " + checkpoint_dir)
print("Saving weights to   -> " + h5_filepath)
model.save_weights(h5_filepath, overwrite=False, save_format='h5')
print("Completed. Goodbye.")