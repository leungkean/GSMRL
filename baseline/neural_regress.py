import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

#GPU setup
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--epoch', type=int)
parser.add_argument('--nhidden', type=int)
args = parser.parse_args()

with open("../data/{}.pkl".format(args.dataset), 'rb') as f: 
    data = pickle.load(f)

def mlp():
    hidden_size = 300 

    m = tf.keras.Sequential() 
    m.add(tf.keras.Input(shape=(data['train'][0].shape[1],))) 
    m.add(tf.keras.layers.Normalization(axis=-1))
    for i in range(args.nhidden):
        m.add(tf.keras.layers.Dense(hidden_size, activation='relu')) 
    m.add(tf.keras.layers.Dense(9, activation='linear'))

    def rmse(y_true, y_pred): 
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    m.compile(loss=rmse, optimizer='Adam')
    m.summary() 

    history = m.fit(data['train'][0], data['train'][1], epochs=args.epoch, 
                    validation_data=(data['valid'][0], data['valid'][1]))
    return (history, m)

def main(): 
    h, model = mlp()

    print("Test:") 
    model.evaluate(data['test'][0], data['test'][1]) 

# If this is the main thread of execution, run the code.
if __name__ == "__main__": 
    main()
