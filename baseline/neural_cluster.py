import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm.keras import TqdmCallback

#GPU setup
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int)
parser.add_argument('--nhidden', type=int)
args = parser.parse_args()

data = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)

cheap_feat = data[:, 5:-8]
cheap_feat[cheap_feat>0] = 1.
cheap_feat[cheap_feat==0] = -1.
exp_feat = data[:,-8:]
cheap_feat = np.array(cheap_feat, np.float32) + np.random.normal(0, 0.01, (data.shape[0], 222)).astype(np.float32)
exp_feat = np.array(data[:,-8:], np.float32).reshape((data.shape[0], 8))
out = data[:, 0]

# Normalize expensive feature to [-1, 1]
for i in range(exp_feat.shape[1]):
    exp_feat[:, i] = 2.*(exp_feat[:, i] - exp_feat[:, i].min()) / np.ptp(exp_feat[:, i]) - 1.
all_feat = np.concatenate((cheap_feat, exp_feat), axis=1)

fold1 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 1]
fold2 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 2]
fold3 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 3]
folds = [fold1, fold2, fold3]

def VGMM(X, k):
    """
    Fitting a Variational Gaussian Mixture Model
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    k : int, optional, default: 8
        Number of clusters to form.
    Returns
    -------
    labels : array, shape (n_samples,)
        Labels of each point.
    """
    bgm = BayesianGaussianMixture(n_components=k, max_iter=1000, verbose=1, covariance_type='full', weight_concentration_prior=0.9, tol=1e-8).fit(X)
    return bgm 

def labels(X, k):
    model = VGMM(X, k)
    labels = model.predict(X)
    weights = model.weights_
    while np.min(weights) < 0.10 or np.min(weights) > 0.21:
        model = VGMM(X, k)
        labels = model.predict(X)
        weights = model.weights_
        print("Search Weights: ", weights)
    print("Weights: ", weights)
    print("Score: ", model.score(X))
    if np.count_nonzero(labels) > 0.5*labels.shape[0]:
        return np.logical_not(labels).astype(np.int32)
    return labels.astype(np.int32)

def all_labels():
    for i in range(3):
        train_exp_idx = labels(all_feat[folds[i] == 0], 2)
        valid_exp_idx = labels(all_feat[folds[i] == 2], 2)
        test_exp_idx = labels(all_feat[folds[i] == 1], 2)
        np.save(f"train_exp_labels{i}.npy", train_exp_idx)
        np.save(f"valid_exp_labels{i}.npy", valid_exp_idx)
        np.save(f"test_exp_labels{i}.npy", test_exp_idx)
    return 

def mlp(train, valid, y_out_train, y_out_valid):
    hidden_size = 256

    m = tf.keras.Sequential() 
    m.add(tf.keras.Input(shape=(train.shape[1],))) 
    for i in range(args.nhidden):
        m.add(tf.keras.layers.Dense(hidden_size, activation='relu')) 
    m.add(tf.keras.layers.Dense(1, activation='linear'))

    def rmse(y_true, y_pred): 
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    m.compile(loss=rmse, optimizer='Adam')

    m.fit(train, y_out_train, epochs=args.epoch, verbose=0,
            validation_data=(valid, y_out_valid), callbacks=[TqdmCallback(verbose=1)])
    return m

def main(relabel): 
    if not os.path.exists("test_exp_labels2.npy") or relabel: 
        all_labels()

    for i in range(3):
        train_exp_idx = np.load(f"train_exp_labels{i}.npy")
        valid_exp_idx = np.load(f"valid_exp_labels{i}.npy")
        test_exp_idx = np.load(f"test_exp_labels{i}.npy")

        train_feat = all_feat[folds[i] == 0]
        train_out = out[folds[i] == 0]
        valid_feat = all_feat[folds[i] == 2]
        valid_out = out[folds[i] == 2]
        test_feat = all_feat[folds[i] == 1]
        test_out = out[folds[i] == 1]

        cheap_feat_train = train_feat[train_exp_idx == 0]
        cheap_feat_valid = valid_feat[valid_exp_idx == 0]
        cheap_feat_test = test_feat[test_exp_idx == 0]
        cheap_out_train = train_out[train_exp_idx == 0]
        cheap_out_valid = valid_out[valid_exp_idx == 0]
        cheap_out_test = test_out[test_exp_idx == 0]

        mask = np.array([1. for _ in range(222)] + [0. for _ in range(8)]).astype(np.float32)
        cheap_feat_train *= mask
        cheap_feat_valid *= mask
        cheap_feat_test *= mask

        exp_feat_train = train_feat[train_exp_idx == 1]
        exp_feat_valid = valid_feat[valid_exp_idx == 1]
        exp_feat_test = test_feat[test_exp_idx == 1]
        exp_out_train = train_out[train_exp_idx == 1]
        exp_out_valid = valid_out[valid_exp_idx == 1]
        exp_out_test = test_out[test_exp_idx == 1]

        print(f"Fold {i+1}:")
        print("Cheap Features Cluster:")
        model = mlp(cheap_feat_train, cheap_feat_valid, cheap_out_train, cheap_out_valid)
        model2 = mlp(train_feat[train_exp_idx == 0], valid_feat[valid_exp_idx == 0], cheap_out_train, cheap_out_valid)
        model.evaluate(cheap_feat_test, cheap_out_test)
        model2.evaluate(exp_feat_test, exp_out_test)

        print("Expensive Features Cluster:")
        model = mlp(exp_feat_train, exp_feat_valid, exp_out_train, exp_out_valid)
        model2 = mlp(exp_feat_train*mask, exp_feat_valid*mask, exp_out_train, exp_out_valid)
        model2.evaluate(cheap_feat_test, cheap_out_test)
        model.evaluate(exp_feat_test, exp_out_test)

# If this is the main thread of execution, run the code.
if __name__ == "__main__": 
    main(True)
