import numpy as np
import pandas as pd
import os
import h5py
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier

class Dataset():

    def __init__(self, path, label_names = ['Disturbed', 'Merging', 'Round Smooth', 'In-between Round Smooth', 'Cigar Shaped Smooth', 'Barred Spiral', 'Unbarred Tight Spiral', 'Unbarred Loose Spiral', 'Edge-on without Bulge', 'Edge-on with Bulge']):
        self.path = path
        self.label_names = label_names
        self.class_dict = {num:gal_type for num,gal_type in zip(range(10), label_names)}
        self.images = None
        self.labels = None

    def load_data(self):
        with h5py.File(self.path, 'r') as F:
            self.images = np.array(F['images']).astype(np.uint8)
            self.labels = np.array(F['ans'])
        return self.images, self.labels

    def plot_few(self, random=False):
        fig, axs = plt.subplots(2,5, figsize=(15,5))

        for idx, plot in enumerate(axs.flatten()):
            class_idx = np.argwhere(self.labels==idx)[0][0]
            plot.imshow(self.images[class_idx])
            plot.set_title(self.class_dict[self.labels[class_idx]])
            plot.axis('off')
        plt.show()

    def scale_data(self, data):
        return data / 255.0

    def prep_data(self, data, resize_shape, crop_idxs = (32,96)):
        low, high = crop_idxs
        resized_ims = tf.image.resize(self.scale_data(data), resize_shape)
        return resized_ims.numpy()[:,low:high, low:high, :]

if __name__=='__main__':
    data = Dataset('data/Galaxy10_DECals.h5')
    images, labels = data.load_data()
    data.plot_few()