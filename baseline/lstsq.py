import tensorflow as tf
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

def main():
    with open("../data/{}.pkl".format(args.dataset), 'rb') as f:
        data = pickle.load(f)

    x_train = data['train'][0]
    y_train = data['train'][1]
    x_valid = data['valid'][0]
    y_valid = data['valid'][1]
    x_test = data['test'][0]
    y_test = data['test'][1]

    train = tf.linalg.lstsq(x_train, y_train, l2_regularizer=0.1)

    y_valid_pred = tf.matmul(x_valid, train)
    valid_rmse = tf.math.sqrt(tf.reduce_mean(tf.square(y_valid_pred - y_valid))).numpy()
    
    y_test_pred = tf.matmul(x_test, train)
    test_rmse = tf.math.sqrt(tf.reduce_mean(tf.square(y_test_pred - y_test))).numpy()

    print("Valid RMSE: {}".format(valid_rmse))
    print("Test RMSE: {}".format(test_rmse))

# Run main
if __name__ == '__main__':
    main()
