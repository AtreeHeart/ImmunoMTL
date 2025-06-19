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
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# === Load Training Data ===
training = pd.read_csv("../data/MHCflurry_training.csv", index_col=0)

# === Perform clustering ===
def kmeans_pred(i, df):
    kmeans = KMeans(n_clusters=i, random_state=30).fit(df)
    model_filename = '../models/kmeans_HLA{}cluster.pkl'.format(i)
    with open(model_filename, 'wb') as file:
        pickle.dump(kmeans, file)

    # Compute cluster centers and predict cluster indices
    clusters = kmeans.predict(df)
    cluster_res = pd.DataFrame(df.index, clusters)
    unique_elements, counts_elements = np.unique(clusters, return_counts=True)
    print("Count of each cluster:")
    print(np.asarray((unique_elements, counts_elements)))

    """Find centroid"""
    tr_array = training.to_numpy()
    for iclust in range(kmeans.n_clusters):
        cluster_pts = tr_array[kmeans.labels_ == iclust]
        cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]
        min_idx = np.argmin([euclidean(tr_array[idx, :], kmeans.cluster_centers_[iclust]) for idx in cluster_pts_indices])
        print('closest index to cluster ' + str(iclust) + ' center: ', training.index[cluster_pts_indices[min_idx]])
    
    return clusters
        
iter_k = pd.DataFrame()
for k in range(3, 7):
    final_pred_c4 = kmeans_pred(i=k, df=training)
    final_df = pd.DataFrame({"HLA":training.index,"cluster": final_pred_c4})
    final_df.to_csv(f"../results/clustering_resC{k}.csv")