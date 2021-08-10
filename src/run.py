import numpy as np
import pandas as pd
import os
import h5py
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy  import DummyClassifier
import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from data import Galaxies
import CNNs
from datetime import datetime

def train_CNN(CNN, save_filename):
    datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

    CNN_aug = CNN.fit(datagen.flow(train_images, train_labels, batch_size=32),
         validation_data=(test_images, test_labels),
         steps_per_epoch=len(train_images) // 32, epochs=30)
    CNN.evaluate(test_images, test_labels)
    
    CNNs.show_final_history(CNN_aug)
    plt.savefig(os.path.join('imgs',save_filename + '.png'))
    plt.show()
    plt.close()

    saved_model_path = os.path.join('saved_models', save_filename + '.h5')
    CNN.save(saved_model_path)
    return CNN_aug

if __name__=='__main__':
    #data = Galaxies('data/Galaxy10_DECals.h5')
    #images, labels = data.load_data()
    #data.plot_few()

    # Split data into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, stratify=labels)

    # Prepare data for neural network ingest
    #train_images, test_images = data.prep_data(X_train, [128,128]), data.prep_data(X_test, [128,128])
    train_images, test_images = np.load('data/lg_train.npy'), np.load('data/lg_test.npy')
    train_labels, test_labels = np.load('data/train_labels.npy'), np.load('data/test_labels.npy')
    # Baseline Classifier
    baseline_clf = DummyClassifier(strategy='most_frequent').fit(train_images, train_labels)
    baseline_score = accuracy_score(test_labels, baseline_clf.predict(test_images))
    print(f'Baseline Accuracy: {baseline_score}')

    # Create neural network
    #CNN = CNNs.create_CNN5((3,3), in_shape=(88,88,3), drop_out=0)
    #CNN_history = CNN.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

    #save_filename = 'CNN5_aug_3x3_00drop_88in_30epochs{}'.format(datetime.now().strftime("%Y%m%d"))
    #CNN_trained = train_CNN(CNN, save_filename)

    # Load saved CNN
    CNN_trained = keras.models.load_model('saved_models/CNN5_aug_3x3_00drop_88in_50epochs20210809.h5')

    # Plot confusion matrix for test data
    CNNs.plot_cm(test_labels, CNN_trained.predict_classes(test_images))