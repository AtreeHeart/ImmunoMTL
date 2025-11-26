import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# === Load original data ===
training = pd.read_csv("../data/MHCflurry_training.csv", index_col=0)
n_clusters = 4

# === Run full clustering ===
full_model = KMeans(n_clusters=n_clusters, random_state=30).fit(training)
full_labels = pd.Series(full_model.labels_, index=training.index)

# === Strategy 1: Leave-One-Out Consistency ===
def leave_one_out_consistency(df, base_labels):
    match_count = 0
    for hla in df.index:
        df_loo = df.drop(index=hla)
        model_loo = KMeans(n_clusters=n_clusters, random_state=30).fit(df_loo)

        # Assign dropped HLA to nearest cluster center
        point = df.loc[[hla]].values
        distances = cdist(point, model_loo.cluster_centers_)
        predicted_cluster = np.argmin(distances)

        if predicted_cluster == base_labels[hla]:
            match_count += 1
    consistency = match_count / len(df)
    return consistency

consistency_score = leave_one_out_consistency(training, full_labels)
print(f"Leave-One-Out Cluster Consistency: {consistency_score:.3f}")

# === Strategy 2: Downsampling Robustness (Bootstrap) ===
def bootstrap_clustering(df, base_labels, n_iter=100, frac=0.9):
    ari_scores = []
    nmi_scores = []

    for i in range(n_iter):
        sample = df.sample(frac=frac, replace=False, random_state=30 + i)
        model = KMeans(n_clusters=n_clusters, random_state=30).fit(sample)
        sample_labels = pd.Series(model.labels_, index=sample.index)

        overlap = sample_labels.index.intersection(base_labels.index)
        ari = adjusted_rand_score(base_labels.loc[overlap], sample_labels.loc[overlap])
        nmi = normalized_mutual_info_score(base_labels.loc[overlap], sample_labels.loc[overlap])
        ari_scores.append(ari)
        nmi_scores.append(nmi)

    return ari_scores, nmi_scores

ari_scores, nmi_scores = bootstrap_clustering(training, full_labels, n_iter=100)

# === Plot Robustness Results ===
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(ari_scores, kde=True, ax=ax[0], color='#7E9AB2')
ax[0].set_title("Adjusted Rand Index (ARI) across bootstraps")
ax[0].set_xlabel("ARI")

sns.histplot(nmi_scores, kde=True, ax=ax[1], color='#F28482')
ax[1].set_title("Normalized Mutual Information (NMI) across bootstraps")
ax[1].set_xlabel("NMI")

plt.savefig(f"../analysis/figures/robustness.png", dpi=300, transparent=True, bbox_inches='tight')
plt.tight_layout()
plt.show()