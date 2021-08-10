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

def create_CNN1(kernel_size = (5,5), in_shape = (64,64,3), drop_out = 0):

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, kernel_size, activation='relu',
                            input_shape=in_shape, padding = "same"),
        keras.layers.MaxPooling2D((2, 2))
    ])
    

    model.add(keras.layers.Conv2D(32, kernel_size, activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation = 'relu'))
    # Output Layer
    model.add(keras.layers.Dense(10,  activation = "softmax"))
    # Compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def create_CNN2(kernel_size = (5,5), in_shape = (64,64,3), drop_out = 0):
    # Layer 1
    model = keras.models.Sequential([
        keras.layers.Conv2D(8, kernel_size, activation='relu',
                            input_shape=in_shape, padding = "same"),
        keras.layers.MaxPooling2D((2, 2))
    ])
    #
    #model.add(keras.layers.Conv2D(8, kernel_size, activation = 'relu', padding = 'same'))
    #model.add(keras.layers.MaxPool2D(2,2))
    
    # Layer 2
    model.add(keras.layers.Conv2D(16, kernel_size, activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPool2D(2,2))
    #model.add(keras.layers.Dropout(drop_out))
    # Layer 3
    model.add(keras.layers.Conv2D(16, kernel_size, activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(keras.layers.Dropout(drop_out))
    model.add(keras.layers.Flatten())
    # Layer 4
    model.add(keras.layers.Dense(64, activation = 'relu'))
    #model.add(keras.layers.Dropout(drop_out))
    # Layer 5
    model.add(keras.layers.Dense(10,  activation = "softmax"))
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def create_CNN5(kernel_size=(3,3), in_shape=(64,64,3), drop_out=0.2, lr = 0.001, l2_reg=0):
    # Layer 1
    model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size, activation='relu',
                        input_shape=in_shape, padding = "same"),
    keras.layers.MaxPooling2D((2, 2))
    ])

    # Layer 2
    model.add(keras.layers.Conv2D(32, kernel_size, activation = 'relu', padding = 'same'))
    model.add(keras.layers.MaxPool2D(2,2))
    #model.add(keras.layers.Dropout(drop_out))
    # Layer 3
    model.add(keras.layers.Conv2D(64, kernel_size, activation = 'relu', padding = 'same', kernel_regularizer = keras.regularizers.l2(l2_reg)))
    model.add(keras.layers.MaxPool2D(2,2))

    model.add(keras.layers.Conv2D(64, kernel_size, activation = 'relu', padding = 'same', kernel_regularizer = keras.regularizers.l2(l2_reg)))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(keras.layers.Dropout(drop_out))
    model.add(keras.layers.Flatten())
    # Layer 4
    model.add(keras.layers.Dense(256, activation = 'relu', kernel_regularizer = keras.regularizers.l2(l2_reg)))
    #model.add(keras.layers.Dropout(drop_out))
    # Layer 5
    model.add(keras.layers.Dense(10,  activation = "softmax"))

    model.compile(optimizer= keras.optimizers.Adam(learning_rate=lr),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

    return model

def show_final_history(history):
    fig, ax = plt.subplots(1,2, figsize=(15,5))

    # Plot learning curves

    # Subplot 1
    ax[0].set_title('Loss')
    ax[0].plot(history.epoch, history.history['loss'], label='Train loss')

    # Subplot 2
    ax[1].set_title('Accuracy')
    ax[1].plot(history.epoch, history.history['accuracy'], label='Train accuracy')

    # Store values that will be used for plotting
    if history.history['val_loss'] and history.history['val_accuracy']:
        min_loss = min(history.history['val_loss'])
        max_acc = max(history.history['val_accuracy'])
        text_xcoord = max(history.epoch)

        ax[0].plot(history.epoch, history.history['val_loss'], label='Validation loss')
        ax[0].axhline(min_loss, color = 'r', label='Min validation value')
        ax[0].annotate(str(np.round(min_loss,3)), (text_xcoord - 1.5, min_loss - 0.05))

        ax[1].plot(history.epoch, history.history['val_accuracy'], label='Validation accuracy')
        ax[1].axhline(max_acc, color = 'r', label='Max validation value')
        ax[1].annotate(str(np.round(max_acc,3)), (text_xcoord - 1.5, max_acc + 0.01))
    
    ax[0].legend(fontsize=12)
    ax[1].legend(fontsize=12)
    #plt.show()

if __name__=='__main__':
    CNN = create_CNN1((3,3))
    CNN.summary()