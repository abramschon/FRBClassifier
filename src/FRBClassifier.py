import random as rnd
import numpy as np
from matplotlib import pyplot as plt
import glob 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from Dataset import Dataset

def main():
    #load data in
    ds = Dataset(prop=0.1)
    train_ds, val_ds, test_ds = ds.get_datasets() #load 10% of the data

    #use model from hpo_model
    model = hpo_model()
    model.summary()

    EPOCHS = 1
    history = model.fit(
        train_ds, 
        epochs = EPOCHS,
        steps_per_epoch = ds.steps_per_epoch, 
        validation_data = val_ds,
    )   

    model.evaluate(test_ds)


def hpo_model():
    """
    Use best architecture from hyper-parameter optimisation
    """
    model = Sequential([
        Conv2D(4, 1, padding='same', activation='relu', input_shape=(244, 244,1)),
        Conv2D(64, 7, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(4, 7, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy',
                                tf.keras.metrics.Precision(), 
                                tf.keras.metrics.Recall()])
    return model           

if __name__ == "__main__":
    main()