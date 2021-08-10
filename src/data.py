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
import random

class Galaxies():

    def __init__(self, path, label_names = ['Disturbed', 'Merging', 'Round Smooth', 'In-between Round Smooth', 'Cigar Shaped Smooth', 'Barred Spiral', 'Unbarred Tight Spiral', 'Unbarred Loose Spiral', 'Edge-on without Bulge', 'Edge-on with Bulge']):
        self.path = path
        self.label_names = label_names
        self.class_dict = {num:gal_type for num,gal_type in zip(range(10), label_names)}
        self.images = None
        self.labels = None

    def count_classes(self, y, ax, xtick_labels=None):
        xtick_labels = self.label_names
        labels, count = np.unique(y, return_counts = True)
        ax.bar(labels, count, tick_label = xtick_labels)
        ax.set_xticklabels(xtick_labels, rotation=60)
        ax.set_title('Class Occurrences in Dataset')
        plt.tight_layout()
        plt.show()
    
    def load_data(self):
        with h5py.File(self.path, 'r') as F:
            self.images = np.array(F['images']).astype(np.uint8)
            self.labels = np.array(F['ans'])
        return self.images, self.labels

    def plot_few(self, rand=False, class_subset = None):
        fig, axs = plt.subplots(2,5, figsize=(15,5))
        if class_subset in self.class_dict:
            print(f'Plotting {self.class_dict[class_subset]} only')
            classes = [class_subset] * 10
        else:
            classes = sorted(self.class_dict.keys())

        for i, (idx, plot) in enumerate(zip(classes, axs.flatten())):
            #print(i, idx)
            class_idxs = np.argwhere(self.labels==idx)
            #print(class_idxs)
            if rand == True:
                plt_idx = random.choice(class_idxs[:,0])
            elif class_subset in self.class_dict:
                plt_idx = class_idxs[i][0]
            else:
                plt_idx = class_idxs[0][0]
            #print(plt_idx)
            plot.imshow(self.images[plt_idx])
            plot.set_title(self.class_dict[self.labels[plt_idx]], fontsize=14)
            plot.axis('off')
        plt.show()

    def scale_data(self, data):
        return data / 255.0

    def prep_data(self, data, resize_shape, crop_idxs = (32,96)):
        low, high = crop_idxs
        resized_ims = tf.image.resize(self.scale_data(data), resize_shape)
        return resized_ims.numpy()[:,low:high, low:high, :]

if __name__=='__main__':
    data = Galaxies('data/Galaxy10_DECals.h5')
    images, labels = data.load_data()
    data.plot_few()
