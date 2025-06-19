# === Development Environment ===
"""
Python version:       3.7.13 
pandas version:       1.3.4
numpy version:        1.21.5
scikit-learn version: 1.0.2
scipy version:        1.7.3
"""

# === Imports ===
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean

# === Load Training Data ===
df = pd.read_csv("../data/MHCflurry_training.csv", index_col=0)

def gmm_pred(df):
    silhouette_scores = []
    components_range = range(3, 10)

    # Evaluate silhouette score to determine optimal number of clusters
    for n in components_range:
        gmm = GaussianMixture(n_components=n, random_state=30).fit(df)
        labels = gmm.predict(df)
        silhouette_scores.append(silhouette_score(df, labels))

    optimal_n = components_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_n}")

    # Final GMM clustering with optimal n
    gmm = GaussianMixture(n_components=optimal_n, random_state=30).fit(df)

    # Save model
    model_filename = f'../models/gmm_HLA{optimal_n}cluster.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(gmm, f)

    # Predict cluster labels
    clusters = gmm.predict(df)
    unique, counts = np.unique(clusters, return_counts=True)
    print("Count of each cluster:")
    print(np.vstack((unique, counts)))

    # Find closest sample to each cluster center
    tr_array = df.to_numpy()
    for c in range(gmm.n_components):
        idxs = np.where(clusters == c)[0]
        distances = [euclidean(tr_array[i], gmm.means_[c]) for i in idxs]
        closest = idxs[np.argmin(distances)]
        print(f"Closest index to cluster {c} center: {df.index[closest]}")

    return pd.DataFrame({'HLA': df.index, 'cluster': clusters})

# Run
gmm_results = gmm_pred(df)
gmm_results.to_csv("../results/gmm_HLA_cluster_assignments.csv", index=False)