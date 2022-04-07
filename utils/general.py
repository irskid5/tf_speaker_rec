import numpy as np
import matplotlib.pyplot as plt
import itertools
import io
import sklearn.metrics
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

def calc_confusion_matrix(y_true, y_pred, num_classes, one_hot):
    if one_hot:
        cm = sklearn.metrics.confusion_matrix(y_true=np.argmax(y_true,axis=-1), y_pred=np.argmax(y_pred, axis=-1), labels=np.arange(num_classes)) 
    else:
        cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=np.arange(num_classes)) 
    return cm

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(len(class_names), len(class_names)))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def save_confusion_matrix(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    # buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    # plt.savefig(buf, format='png')
    plt.savefig("cm.png", dpi=100)
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    # buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    # image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    # image = tf.expand_dims(image, 0)
    
    return None

def confusion_matrix(y_true, y_pred, num_classes, one_hot=True):
    cm = calc_confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=num_classes, one_hot=one_hot)
    class_str_labels = [str(x) for x in range(num_classes)]
    fig = plot_confusion_matrix(cm, class_str_labels)
    save_confusion_matrix(fig)
    return None