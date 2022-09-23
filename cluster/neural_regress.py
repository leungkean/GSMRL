import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
import tensorflow_probability as tfp

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int)
parser.add_argument('--nhidden', type=int)
args = parser.parse_args()

fold = 1

features = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 5:]
cheap_feat = features[:,:-8]
cheap_feat[cheap_feat>0] = 1.
cheap_feat[cheap_feat==0] = -1.
exp_feat = features[:,-8:]
cheap_feat = np.array(cheap_feat, np.float32) + np.random.normal(0, 0.01, (features.shape[0], 222)).astype(np.float32)
exp_feat = np.array(features[:,-8:], np.float32).reshape((features.shape[0], 8))
# Normalize expensive feature to [-1, 1]
for i in range(exp_feat.shape[1]):
    exp_feat[:, i] = 2.*(exp_feat[:, i] - exp_feat[:, i].min()) / np.ptp(exp_feat[:, i]) - 1.
data = np.concatenate((cheap_feat, exp_feat), axis=1)

labels = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 0]
labels = labels.reshape(-1, 1)

indices = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, fold]
train_indices = indices == 0
valid_indices = indices == 2
test_indices = indices == 1
train_inputs = data[train_indices, :]
train_labels = labels[train_indices]
valid_inputs = data[valid_indices, :]
valid_labels = labels[valid_indices]
test_labels = labels[test_indices]
test_inputs = data[test_indices, :]

def mlp():
    hidden_size = 300 

    inp = tf.keras.Input(shape=(train_inputs.shape[1],)) 
    x = tf.keras.layers.Dense(hidden_size, activation='relu')(inp)
    for i in range(args.nhidden-1):
        x = tf.keras.layers.Dense(hidden_size, activation='relu')(x)
    out = tf.keras.layers.Dense(1)(x)

    m = tf.keras.Model(inp, out)

    def rmse(y_true, y_pred): 
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    m.compile(loss=rmse, optimizer='Adam')
    m.summary() 

    history = m.fit(train_inputs, train_labels, epochs=args.epoch, 
                    validation_data=(valid_inputs, valid_labels))
    return (history, m)

def main(): 
    h, model = mlp()

    print("Test:") 
    model.evaluate(train_inputs, train_labels) 
    print(model.predict(train_inputs))

# If this is the main thread of execution, run the code.
if __name__ == "__main__": 
    main()
