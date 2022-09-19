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
parser.add_argument('--nu', type=int)
args = parser.parse_args()

def load_data(full, fold=1):
    features = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 5:]
    cheap_feat = features[:,:-8]
    cheap_feat[cheap_feat>0] = 1.
    cheap_feat[cheap_feat==0] = -1.
    exp_feat = features[:,-8:]
    cheap_feat = np.array(cheap_feat, np.float32) + np.random.normal(0, 0.01, (features.shape[0], 222)).astype(np.float32)
    #top_20 = top_20 = [127,  45, 193, 103, 116, 142, 138, 197,  92,   2,  93, 150, 129, 22,  60, 143, 115, 118,  15, 181]
    #top_20.sort()
    #cheap_feat = cheap_feat[:, top_20]
    exp_feat = np.array(features[:,-8:], np.float32).reshape((features.shape[0], 8))
    # Normalize expensive feature to [-1, 1]
    for i in range(exp_feat.shape[1]):
        exp_feat[:, i] = 2.*(exp_feat[:, i] - exp_feat[:, i].min()) / np.ptp(exp_feat[:, i]) - 1.
    if full: 
        data = np.concatenate((cheap_feat, exp_feat), axis=1)
    else:
        data = cheap_feat
    labels = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 0]
    labels = labels.reshape(-1, 1)
    labels = labels.astype(np.float32)
    indices = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, fold]
    train_inputs = data[indices==0, :]
    train_labels = labels[indices==0]
    valid_inputs = data[indices==2, :]
    valid_labels = labels[indices==2]
    test_labels = labels[indices==1]
    test_inputs = data[indices==1, :]
    return (train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels)

def mix_layer(x):
    mu = tf.expand_dims(tf.math.abs(x[:, 0]), axis=1)
    sigma = tf.expand_dims(tf.math.abs(x[:, 1]), axis=1)
    return tf.concat([mu, sigma], axis=1)

def mlp(train_inputs, train_labels, valid_inputs, valid_labels):
    hidden_size = 256 

    m = tf.keras.Sequential()
    m.add(tf.keras.Input(shape=(train_inputs.shape[1],)))
    for i in range(args.nhidden): 
        m.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
    m.add(tf.keras.layers.Dense(2))
    m.add(tf.keras.layers.Lambda(mix_layer))

    def normal_loss(y_true, y_pred): 
        mu = tf.expand_dims(y_pred[:, 0], axis=1)
        sigma = tf.expand_dims(y_pred[:, 1], axis=1)
        dist = -0.5*tf.math.log(2*np.pi*sigma**2) - 0.5*(y_true - mu)**2/sigma**2 
        return tf.math.reduce_mean(-dist)

    m.compile(loss=normal_loss, optimizer='Adam')
    m.summary() 

    history = m.fit(train_inputs, train_labels, epochs=args.epoch, 
                    validation_data=(valid_inputs, valid_labels))
    return (history, m)

train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels = load_data(bool(args.nu))
h, model = mlp(train_inputs, train_labels, valid_inputs, valid_labels)
model.save(f'prob{args.nu}')

print("Test:") 
model.evaluate(test_inputs, test_labels)
predict = model.predict(test_inputs)
print(predict)
plt.figure()
#plt.scatter(test_labels, predict[:, 0], marker='.')
plt.errorbar(test_labels, predict[:, 0], yerr=predict[:, 1], fmt='o', ecolor='r')
plt.plot(np.linspace(220, 1100, 100), np.linspace(220, 1100, 100), '-k')
plt.xlabel("True value")
plt.ylabel("Predicted value")
if args.nu == 0:
    plt.title("MLP on cheap features")
elif args.nu == 1:
    plt.title("MLP on expensive features")
plt.show()
