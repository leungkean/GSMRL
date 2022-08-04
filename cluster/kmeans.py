from sklearn.cluster import KMeans
import numpy as np

def kmeans(X, k):
    """
    K-means clustering.
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
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    return (kmeans.labels_, kmeans.n_iter_, kmeans.n_features_in_)

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
    train_indices = np.genfromtxt("../../chemistry/dyes_cheap_expensive_with_folds.csv", delimiter=',', skip_header=1, dtype=float)[:, 1] == 0
    data = data[train_indices, :]

    labels, n_iter, feat_in = kmeans(data, 2)

    """
    iterations = 0
    while np.count_nonzero(labels) > data.shape[0]*0.1:
        labels, n_iter, feat_in = kmeans(data, 2)
        iterations += 1
        if (iterations+1) % 10 == 0:
            print("Iteration:", iterations)
            print("z = 1:", np.count_nonzero(labels))
    """

    print("Number of iterations:", n_iter)
    print("Number of features in:", feat_in)
    print("Labels:", labels)
