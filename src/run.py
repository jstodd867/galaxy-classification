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
from data import Dataset
import CNNs

if __name__=='__main__':
    data = Dataset('data/Galaxy10_DECals.h5')
    images, labels = data.load_data()
    #data.plot_few()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, stratify=labels)

    # Prepare data for neural network ingest
    train_images, test_images = data.prep_data(X_train, [128,128]), data.prep_data(X_test, [128,128])

    # Baseline Classifier
    baseline_clf = DummyClassifier(strategy='most_frequent').fit(train_images, y_train)
    baseline_score = accuracy_score(y_test, baseline_clf.predict(test_images))
    print(f'Baseline Accuracy: {baseline_score}')

    # Create neural network
    CNN = CNNs.create_CNN5(drop_out=0)

    datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

    CNN5_aug = CNN.fit(datagen.flow(train_images, y_train, batch_size=32),
         validation_data=(test_images, y_test),
         steps_per_epoch=len(train_images) // 32, epochs=50)

    CNNs.show_final_history(CNN5_aug)