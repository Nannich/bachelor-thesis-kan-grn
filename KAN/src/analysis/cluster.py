import numpy as np
from sklearn.cluster import KMeans

def cluster_trajectories(trajectory_cube, n_clusters=15, lineage=None):
    """
    Clusters genes based on their smoothed trajectories.
    """
    n_genes, n_lineages, n_bins = trajectory_cube.shape

    if lineage is not None:
        # Cluster based on a specific lineage -> Shape: (n_genes, n_bins)
        X = trajectory_cube[:, lineage, :]
    else:
        # Cluster based on all lineages combined -> Shape: (n_genes, n_lineages * n_bins)
        X = trajectory_cube.reshape(n_genes, n_lineages * n_bins)

    # Standardize each genes trajectory (Z-score normalization)
    # This ensures genes with similar expression shapes cluster together 
    # regardless of differences in absolute expression levels
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True) + 1e-8
    X_scaled = (X - X_mean) / X_std

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X_scaled)

    return labels