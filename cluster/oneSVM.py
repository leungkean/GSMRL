from sklearn import svm
from sklearn.neural_network import MLPRegressor
import numpy as np

features = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 5:]
label = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 0]
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
label = label.astype(np.float32)

fold0 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 1]
fold1 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 2]
fold2 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 3]
fold3 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 4]

folds = [fold0, fold1, fold2, fold3]

train0 = data[fold0 == 0, :]
train1 = data[fold1 == 0, :]
train2 = data[fold2 == 0, :]
train3 = data[fold3 == 0, :]

train0_label = label[fold0 == 0]
train1_label = label[fold1 == 0]
train2_label = label[fold2 == 0]
train3_label = label[fold3 == 0]

train = [train0, train1, train2, train3]
train_labels = [train0_label, train1_label, train2_label, train3_label]

test0 = data[fold0 == 1, :]
test1 = data[fold1 == 1, :]
test2 = data[fold2 == 1, :]
test3 = data[fold3 == 1, :]

test0_label = label[fold0 == 1]
test1_label = label[fold1 == 1]
test2_label = label[fold2 == 1]
test3_label = label[fold3 == 1]

test = [test0, test1, test2, test3]
test_labels = [test0_label, test1_label, test2_label, test3_label]

for i in range(len(test)): 
    print(f"Fold {i}:")
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", cache_size=3000, verbose=True, tol=1e-8).fit(test[i])

    # One Class SVM
    pred = clf.predict(test[i])
    weight = [np.count_nonzero(pred == 1)/pred.shape[0], np.count_nonzero(pred == -1)/pred.shape[0]]
    print(f"One Class SVM:")
    print("Weight:", weight)
    print("Score:", np.mean(clf.score_samples(test[i])))

    print("Main Features")
    model_cheap = MLPRegressor(hidden_layer_sizes=(300,300,300), max_iter=5000, n_iter_no_change=25) 
    cheap_test = test[i][pred == 1] * np.array([1. for _ in range(222)] + [0. for _ in range(8)])
    model_cheap.fit(cheap_test, test_labels[i][pred == 1]) 
    print(f"Cheap Dataset:", np.sqrt(np.mean((model_cheap.predict(cheap_test) - test_labels[i][pred == 1])**2)))
    model_full = MLPRegressor(hidden_layer_sizes=(300,300,300), max_iter=5000, n_iter_no_change=25) 
    model_full.fit(test[i][pred == 1], test_labels[i][pred == 1]) 
    print(f"Full Dataset:", np.sqrt(np.mean((model_full.predict(test[i][pred == 1]) - test_labels[i][pred == 1])**2)))

    print("Outlier Features")
    model_cheap = MLPRegressor(hidden_layer_sizes=(300,300,300), max_iter=5000, n_iter_no_change=25) 
    cheap_test = test[i][pred == -1] * np.array([1. for _ in range(222)] + [0. for _ in range(8)])
    model_cheap.fit(cheap_test, test_labels[i][pred == -1]) 
    print(f"Cheap Dataset:", np.sqrt(np.mean((model_cheap.predict(cheap_test) - test_labels[i][pred == -1])**2)))
    model_full = MLPRegressor(hidden_layer_sizes=(300,300,300), max_iter=5000, n_iter_no_change=25) 
    model_full.fit(test[i][pred == -1], test_labels[i][pred == -1]) 
    print(f"Full Dataset:", np.sqrt(np.mean((model_full.predict(test[i][pred == -1]) - test_labels[i][pred == -1])**2)))
