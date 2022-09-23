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
    bgm = BayesianGaussianMixture(n_components=k, max_iter=1000, verbose=1, covariance_type='full', weight_concentration_prior=0.9, tol=1e-8).fit(X)
    return bgm 

def train(fold):
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

    if fold == 1: 
        fold1 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 1]
        train = data[fold1 == 0, :]
    elif fold == 2: 
        fold2 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 2]
        train = data[fold2 == 0, :]
    else: 
        fold3 = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 3]
        train = data[fold3 == 0, :]

    model = VGMM(train, k=2)
    score = model.score(train)
    weights = model.weights_
    '''
    while np.min(weights) < 0.09 or np.min(weights) > 0.15:
        model = VGMM(train, k=2)
        weights = model.weights_
        print(weights)
    '''
    labels = model.predict(train)

    print("Training")
    print("Means:", model.means_)
    print("Labels:", labels)
    print("Score:", score)
    print("Weights:", weights)

    return model, labels

train(1)
