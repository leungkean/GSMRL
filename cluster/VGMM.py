from sklearn.mixture import BayesianGaussianMixture
import numpy as np

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
    bgm = BayesianGaussianMixture(n_components=k, max_iter=1000, verbose=1, covariance_type='tied', weight_concentration_prior=0.9, tol=1e-6, init_params='random_from_data').fit(X)
    return bgm 

if __name__ == '__main__':
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
    out = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 0]
    out = out.reshape(-1, 1)
    #data = np.concatenate((data, out), axis=1)

    train_indices = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 1] == 0
    train = data[train_indices, :]

    model = VGMM(train, k=2)
    score = model.score(train)
    weights = model.weights_
    while np.min(weights) < 0.1 or np.min(weights) > 0.15:
        model = VGMM(train, k=2)
        weights = model.weights_
        print(weights)
    labels = model.predict(train)

    print("Training")
    print("Means:", model.means_)
    print("Labels:", labels)
    print("Score:", score)
    print("Weights:", weights)

    np.save("exp_train.npy", labels)

    test_indices = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 1] == 1
    test = data[test_indices, :]

    labels = model.predict(test)
    score = model.score(test)
    weights = model.weights_
    print("Testing")
    print("Labels:", labels)
    print("Score:", score)
    print("Weights:", weights)

    np.save("exp_test.npy", labels)
