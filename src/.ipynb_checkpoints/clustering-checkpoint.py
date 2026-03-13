import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

def cluster_pixels(pixels, algorithm='kmeans', k=3, **kwargs):
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=k, n_init=5, **kwargs)
    elif algorithm == 'gmm':
        model = GaussianMixture(n_components=k, **kwargs)
    elif algorithm == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=k, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'kmeans', 'gmm', 'agglomerative'")

    if algorithm == 'agglomerative':
        model.fit(pixels)
        labels = model.labels_
    else:
        model.fit(pixels)
        labels = model.predict(pixels)

    centers = np.array([pixels[labels == i].mean(axis=0) for i in range(k)]) # computed consistently across all algorithms for comparability
    
    return labels, centers