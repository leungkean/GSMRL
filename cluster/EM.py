from sklearn.mixture import GaussianMixture
import numpy as np

def GMM(X, k):
    """
    Fitting a Gaussian Mixture Model with EM
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
    gmm = GaussianMixture(n_components=k, tol=1e-8, weights_init=[0.9, 0.1], covariance_type='full', verbose=1).fit(X)
    return gmm

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
    data = data[train_indices, :]

    model = GMM(data, k=2)
    labels = model.predict(data)

    print("Means:", model.means_)
    print("Labels:", labels[:100])
    print("Weights:", model.weights_)
    print("Score:", model.score(data))
