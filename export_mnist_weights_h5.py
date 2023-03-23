import os

import tensorflow as tf
from tensorflow import keras

from mnist_rnn_model import get_model

pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202303/20230309-102046/checkpoints/"

model = get_model()

if pretrained_weights is not None:
    model.load_weights(pretrained_weights)
    print('Restored pretrained weights from {}.'.format(pretrained_weights))

# Reset the stat variables
weights = model.get_weights()
weights_to_discard = []
for i in range(len(weights)):
    if "/w" in model.weights[i].name or "/x" in model.weights[i].name:
        weights[i] = 0*weights[i]
model.set_weights(weights)

h5_dir = pretrained_weights + "hdf5/"
if not os.path.exists(h5_dir):
    os.makedirs(h5_dir)
h5_filepath = h5_dir + "weights.hdf5"

print("Saving weights from -> " + pretrained_weights)
print("Saving weights to   -> " + h5_filepath)
model.save_weights(h5_filepath, overwrite=False, save_format='h5')
print("Completed. Goodbye.")