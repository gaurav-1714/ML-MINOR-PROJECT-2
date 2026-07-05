"""
Minor Project 2 - Customer Segmentation using Unsupervised Machine Learning
Pipeline: Data Generation -> EDA -> Preprocessing -> K-Means -> Hierarchical -> DBSCAN -> Evaluation
"""

import json
import warnings
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
np.random.seed(42)

DATA_DIR   = "data"
MODEL_DIR  = "model"
RESULTS_DIR = "results"

# ─────────────────────────────────────────────
# 1. GENERATE / LOAD DATASET
# ─────────────────────────────────────────────
print("### SECTION 1: DATASET ###")

# Reproduce the well-known Mall Customer Segmentation dataset structure
# (200 customers, same distributional properties as the Kaggle dataset)
n = 200
np.random.seed(42)

customer_id = np.arange(1, n + 1)
gender      = np.random.choice(["Male", "Female"], size=n, p=[0.44, 0.56])

# 5 natural customer segments replicated from the real dataset
# Segment A: young, low income, high spending
# Segment B: young, high income, high spending
# Segment C: middle age, medium income, medium spending
# Segment D: old, high income, low spending
# Segment E: middle age, low income, low spending

seg_sizes = [40, 35, 60, 35, 30]
ages, incomes, scores = [], [], []

params = [
    # (age_mean, age_std, income_mean, income_std, score_mean, score_std)
    (25, 4,  25, 6,  77, 10),   # A - young, low income, high spending
    (28, 5,  85, 10, 82, 8),    # B - young, high income, high spending
    (42, 8,  55, 12, 50, 13),   # C - middle, mid income, mid spending
    (50, 9,  87, 10, 18, 8),    # D - older, high income, low spending
    (48, 9,  26, 7,  20, 9),    # E - older, low income, low spending
]

for (am, astd, im, istd, sm, sstd), sz in zip(params, seg_sizes):
    ages.append(np.clip(np.random.normal(am, astd, sz), 18, 70).astype(int))
    incomes.append(np.clip(np.random.normal(im, istd, sz), 15, 137).astype(int))
    scores.append(np.clip(np.random.normal(sm, sstd, sz), 1, 99).astype(int))

ages    = np.concatenate(ages)
incomes = np.concatenate(incomes)
scores  = np.concatenate(scores)

# Shuffle
idx = np.random.permutation(n)
ages, incomes, scores, gender = ages[idx], incomes[idx], scores[idx], gender[idx]

df = pd.DataFrame({
    "CustomerID":        customer_id,
    "Gender":            gender,
    "Age":               ages,
    "Annual_Income_kUSD": incomes,
    "Spending_Score":    scores
})

print("Shape:", df.shape)
print(df.head(8).to_string())
df.to_csv(f"{DATA_DIR}/mall_customers.csv", index=False)
print(f"\nSaved {DATA_DIR}/mall_customers.csv")

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n### SECTION 2: EDA ###")

print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())
print("\nGender distribution:\n", df["Gender"].value_counts().to_string())
print("\nDescriptive stats:")
print(df[["Age","Annual_Income_kUSD","Spending_Score"]].describe().to_string())

# Plot 1 – Gender distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
gender_counts = df["Gender"].value_counts()
axes[0].pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",
            colors=["#5B8FDB","#F4889A"], startangle=90)
axes[0].set_title("Gender Distribution")
sns.countplot(data=df, x="Gender", palette=["#5B8FDB","#F4889A"], ax=axes[1])
axes[1].set_title("Gender Count")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/eda_gender_distribution.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/eda_gender_distribution.png")

# Plot 2 – Feature distributions
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
features = ["Age", "Annual_Income_kUSD", "Spending_Score"]
colors   = ["#5B8FDB", "#F4889A", "#5DBE8A"]
for ax, feat, col in zip(axes, features, colors):
    sns.histplot(df[feat], kde=True, ax=ax, color=col, bins=20)
    ax.set_title(f"Distribution of {feat.replace('_',' ')}")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/eda_feature_distributions.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/eda_feature_distributions.png")

# Plot 3 – Pair scatter plots (Income vs Spending coloured by Gender)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (x, y) in zip(axes, [("Age","Spending_Score"), ("Annual_Income_kUSD","Spending_Score")]):
    for g, col in [("Male","#5B8FDB"), ("Female","#F4889A")]:
        sub = df[df["Gender"]==g]
        ax.scatter(sub[x], sub[y], label=g, alpha=0.6, color=col, s=50)
    ax.set_xlabel(x.replace("_"," "))
    ax.set_ylabel("Spending Score")
    ax.set_title(f"{x.replace('_',' ')} vs Spending Score")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/eda_scatter_plots.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/eda_scatter_plots.png")

# Plot 4 – Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/eda_correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/eda_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n### SECTION 3: PREPROCESSING ###")

# Encode Gender (for potential use)
df["Gender_enc"] = (df["Gender"] == "Female").astype(int)

# Work on the 3 numeric features used for clustering
X = df[["Age", "Annual_Income_kUSD", "Spending_Score"]].values
X2 = df[["Annual_Income_kUSD", "Spending_Score"]].values   # 2D for easy visualisation

scaler2 = StandardScaler()
scaler3 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)
X3_scaled = scaler3.fit_transform(X)

joblib.dump(scaler3, f"{MODEL_DIR}/scaler_3d.pkl")
joblib.dump(scaler2, f"{MODEL_DIR}/scaler_2d.pkl")
print("StandardScaler fitted and saved.")

# ─────────────────────────────────────────────
# 4a. K-MEANS — Elbow + Silhouette to find optimal k
# ─────────────────────────────────────────────
print("\n### SECTION 4a: K-MEANS ###")

inertias, sil_scores, db_scores, ch_scores = [], [], [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(X3_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X3_scaled, labels))
    db_scores.append(davies_bouldin_score(X3_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X3_scaled, labels))

# Elbow + Silhouette plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), inertias, "bo-", lw=2)
axes[0].axvline(x=5, color="red", linestyle="--", label="k=5 (elbow)")
axes[0].set_xlabel("Number of Clusters (k)"); axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method"); axes[0].legend()

axes[1].plot(list(K_range), sil_scores, "gs-", lw=2)
axes[1].axvline(x=5, color="red", linestyle="--", label="k=5 (best sil)")
axes[1].set_xlabel("Number of Clusters (k)"); axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs k"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/kmeans_elbow_silhouette.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/kmeans_elbow_silhouette.png")

BEST_K = 5
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=20)
km_labels = km_final.fit_predict(X3_scaled)
df["KMeans_Cluster"] = km_labels

km_sil = silhouette_score(X3_scaled, km_labels)
km_db  = davies_bouldin_score(X3_scaled, km_labels)
km_ch  = calinski_harabasz_score(X3_scaled, km_labels)
print(f"K-Means (k=5): Silhouette={km_sil:.4f}, DB={km_db:.4f}, CH={km_ch:.2f}")

joblib.dump(km_final, f"{MODEL_DIR}/kmeans_k5.pkl")

# Cluster visualisation – Income vs Spending (most interpretable 2D view)
palette = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"]
fig, ax = plt.subplots(figsize=(8, 6))
for c in range(BEST_K):
    mask = km_labels == c
    ax.scatter(df.loc[mask,"Annual_Income_kUSD"], df.loc[mask,"Spending_Score"],
               label=f"Cluster {c+1}", color=palette[c], s=60, alpha=0.8)
# plot centroids back in original space
centroids_orig = scaler3.inverse_transform(km_final.cluster_centers_)
ax.scatter(centroids_orig[:,1], centroids_orig[:,2],
           c="black", marker="X", s=200, zorder=5, label="Centroids")
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score")
ax.set_title("K-Means Clustering (k=5): Income vs Spending Score")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/kmeans_cluster_income_spending.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/kmeans_cluster_income_spending.png")

# PCA 2D projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X3_scaled)
fig, ax = plt.subplots(figsize=(7, 5))
for c in range(BEST_K):
    mask = km_labels == c
    ax.scatter(X_pca[mask,0], X_pca[mask,1],
               label=f"Cluster {c+1}", color=palette[c], s=60, alpha=0.8)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("K-Means Clusters (PCA 2D Projection)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/kmeans_pca_projection.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/kmeans_pca_projection.png")

# Cluster profile table
profile = df.groupby("KMeans_Cluster")[["Age","Annual_Income_kUSD","Spending_Score"]].mean().round(1)
profile["Size"] = df["KMeans_Cluster"].value_counts().sort_index()
print("\nK-Means Cluster Profiles:")
print(profile.to_string())

# ─────────────────────────────────────────────
# 4b. HIERARCHICAL CLUSTERING
# ─────────────────────────────────────────────
print("\n### SECTION 4b: HIERARCHICAL CLUSTERING ###")

Z = linkage(X3_scaled, method="ward")

# Dendrogram
fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_font_size=9, ax=ax,
           color_threshold=0.7 * max(Z[:,2]))
ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
ax.set_xlabel("Sample Index (or cluster size)")
ax.set_ylabel("Ward Distance")
ax.axhline(y=6.5, color="red", linestyle="--", label="Cut at 5 clusters")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/hierarchical_dendrogram.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/hierarchical_dendrogram.png")

hc = AgglomerativeClustering(n_clusters=5, linkage="ward")
hc_labels = hc.fit_predict(X3_scaled)
df["HC_Cluster"] = hc_labels

hc_sil = silhouette_score(X3_scaled, hc_labels)
hc_db  = davies_bouldin_score(X3_scaled, hc_labels)
hc_ch  = calinski_harabasz_score(X3_scaled, hc_labels)
print(f"Hierarchical (k=5): Silhouette={hc_sil:.4f}, DB={hc_db:.4f}, CH={hc_ch:.2f}")

joblib.dump(hc, f"{MODEL_DIR}/hierarchical_k5.pkl")

# Visualisation
fig, ax = plt.subplots(figsize=(8, 6))
for c in range(5):
    mask = hc_labels == c
    ax.scatter(df.loc[mask,"Annual_Income_kUSD"], df.loc[mask,"Spending_Score"],
               label=f"Cluster {c+1}", color=palette[c], s=60, alpha=0.8)
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score")
ax.set_title("Hierarchical Clustering (Ward, k=5): Income vs Spending")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/hierarchical_cluster_income_spending.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/hierarchical_cluster_income_spending.png")

# ─────────────────────────────────────────────
# 4c. DBSCAN
# ─────────────────────────────────────────────
print("\n### SECTION 4c: DBSCAN ###")

# Tune eps using k-distance graph (k=5)
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X3_scaled)
dists, _ = nbrs.kneighbors(X3_scaled)
k_dists = np.sort(dists[:, -1])[::-1]

plt.figure(figsize=(7, 4))
plt.plot(k_dists, color="#3498db", lw=2)
plt.axhline(y=0.65, color="red", linestyle="--", label="eps ≈ 0.65")
plt.xlabel("Sorted Sample Index"); plt.ylabel("5-NN Distance")
plt.title("K-Distance Graph (k=5) — Choosing eps for DBSCAN")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/dbscan_kdistance.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/dbscan_kdistance.png")

db = DBSCAN(eps=0.65, min_samples=5)
db_labels = db.fit_predict(X3_scaled)
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = int((db_labels == -1).sum())
print(f"DBSCAN: {n_clusters_db} clusters found, {n_noise} noise points")

joblib.dump(db, f"{MODEL_DIR}/dbscan.pkl")

# Only score if we have ≥2 clusters and not all noise
if n_clusters_db >= 2 and n_noise < n:
    mask_valid = db_labels != -1
    db_sil = silhouette_score(X3_scaled[mask_valid], db_labels[mask_valid])
    db_db  = davies_bouldin_score(X3_scaled[mask_valid], db_labels[mask_valid])
    db_ch  = calinski_harabasz_score(X3_scaled[mask_valid], db_labels[mask_valid])
    print(f"DBSCAN (valid points only): Silhouette={db_sil:.4f}, DB={db_db:.4f}, CH={db_ch:.2f}")
else:
    db_sil, db_db, db_ch = None, None, None
    print("DBSCAN could not form ≥2 clusters with chosen eps/min_samples.")

# DBSCAN plot
fig, ax = plt.subplots(figsize=(8, 6))
unique_labels = sorted(set(db_labels))
db_palette = ["#cccccc"] + ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
                              "#1abc9c","#e67e22","#8e44ad"]
for i, lbl in enumerate(unique_labels):
    mask = db_labels == lbl
    color = db_palette[0] if lbl == -1 else db_palette[i]
    label = "Noise" if lbl == -1 else f"Cluster {lbl+1}"
    ax.scatter(df.loc[mask,"Annual_Income_kUSD"], df.loc[mask,"Spending_Score"],
               label=label, color=color, s=60, alpha=0.8)
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score")
ax.set_title("DBSCAN Clustering: Income vs Spending Score")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/dbscan_cluster_income_spending.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/dbscan_cluster_income_spending.png")

# ─────────────────────────────────────────────
# 5. COMPARISON & SUMMARY
# ─────────────────────────────────────────────
print("\n### SECTION 5: MODEL COMPARISON ###")

summary = {
    "K-Means (k=5)": {
        "n_clusters": 5,
        "silhouette": round(km_sil, 4),
        "davies_bouldin": round(km_db, 4),
        "calinski_harabasz": round(km_ch, 2),
    },
    "Hierarchical (Ward, k=5)": {
        "n_clusters": 5,
        "silhouette": round(hc_sil, 4),
        "davies_bouldin": round(hc_db, 4),
        "calinski_harabasz": round(hc_ch, 2),
    },
    "DBSCAN (eps=0.65, min=5)": {
        "n_clusters": n_clusters_db,
        "noise_points": n_noise,
        "silhouette": round(db_sil, 4) if db_sil else "N/A",
        "davies_bouldin": round(db_db, 4) if db_db else "N/A",
        "calinski_harabasz": round(db_ch, 2) if db_ch else "N/A",
    },
}

for name, m in summary.items():
    print(f"\n{name}: {m}")

# Bar chart comparison (K-Means and Hierarchical only for bar chart since both numeric)
comp_data = {
    "Model": ["K-Means", "Hierarchical"],
    "Silhouette ↑": [km_sil, hc_sil],
    "Davies-Bouldin ↓": [km_db, hc_db],
}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(2)
axes[0].bar(x, comp_data["Silhouette ↑"], color=["#3498db","#2ecc71"], width=0.4)
axes[0].set_xticks(x); axes[0].set_xticklabels(comp_data["Model"])
axes[0].set_title("Silhouette Score (higher = better)")
axes[0].set_ylim(0, 0.8)
for i, v in enumerate(comp_data["Silhouette ↑"]):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

axes[1].bar(x, comp_data["Davies-Bouldin ↓"], color=["#3498db","#2ecc71"], width=0.4)
axes[1].set_xticks(x); axes[1].set_xticklabels(comp_data["Model"])
axes[1].set_title("Davies-Bouldin Index (lower = better)")
for i, v in enumerate(comp_data["Davies-Bouldin ↓"]):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
plt.close()
print(f"\nSaved {RESULTS_DIR}/model_comparison.png")

# Cluster profiles (K-Means) – save to CSV
profile.to_csv(f"{RESULTS_DIR}/kmeans_cluster_profiles.csv")
print(f"Saved {RESULTS_DIR}/kmeans_cluster_profiles.csv")

with open(f"{RESULTS_DIR}/metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(f"{RESULTS_DIR}/metrics_summary.txt", "w") as f:
    f.write("CLUSTERING EVALUATION SUMMARY\n" + "="*40 + "\n\n")
    for name, m in summary.items():
        f.write(f"{name}\n")
        for k, v in m.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
print(f"Saved {RESULTS_DIR}/metrics_summary.json and .txt")

print("\n### PIPELINE COMPLETE ###")
