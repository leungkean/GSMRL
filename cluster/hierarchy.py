from sklearn.cluster import AgglomerativeClustering
import numpy as np

def clustering(X, n_clusters):
    """
    Clustering function
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', verbose=1)
    clustering.fit(X)
    return (clustering.labels_, clustering.n_features_in_, clustering.distances_)


