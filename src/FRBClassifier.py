import random as rnd
import numpy as np
from matplotlib import pyplot as plt
import glob 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Flatten, Dropout, Dense, GlobalAveragePooling2D, BatchNormalization

from Dataset import Dataset

def main():
    #load data in
    ds = Dataset(prop=0.1)
    train_ds, val_ds, test_ds = ds.get_datasets() #load 10% of the data

    #use model from hpo_model
    reg = reg_model()
    reg.summary()

    reg.evaluate(test_ds.take(5))

    depth = depth_model(2)
    depth.summary()

    depth.evaluate(test_ds.take(5))

def depth_model(depth, input_shape=(244,244,1)):
    model = Sequential()

    #1st hidden 'block'
    model.add(Conv2D(16, 3, padding='same', input_shape=input_shape, activation='relu')) 
    model.add(Conv2D(16, 3, padding='same', activation='relu')) 
    model.add(MaxPooling2D())
    
    #2 or more hidden 'blocks' depending on depth
    for i in range(1, depth):
        model.add(Conv2D(16*2**i, 3, padding='same', activation='relu')) 
        model.add(Conv2D(16*2**i, 3, padding='same', activation='relu')) 
        if i < (depth-1):
            model.add(MaxPooling2D())

    #finally, global average pool and sigmoidal classificaiton
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.BinaryCrossentropy(), 
        metrics = ['accuracy',
                    tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall()])

    return model
     

def reg_model(input_shape=(244,244,1)):
    model = Sequential([
        Conv2D(16, 3, padding='same', input_shape=input_shape), 
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),
        Dropout(0.2, (None, 1,1,None)), #drop channel
        Conv2D(32, 3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.2), #drop neuron
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy',
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall()])
    return model   


if __name__ == "__main__":
    main()