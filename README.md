# Minor Project 2: Customer Segmentation using Unsupervised Machine Learning

An unsupervised machine learning project that groups retail customers into meaningful segments using K-Means, Hierarchical Clustering, and DBSCAN — based on their Age, Annual Income, and Spending Score.

---

## 1. Problem Statement

Retail businesses serve customers with widely varying demographics and spending behaviours. Without labels or categories, it is difficult to design personalised marketing campaigns or loyalty programmes. This project applies **unsupervised machine learning** to automatically discover distinct customer groups from raw demographic and behavioural data — with no pre-assigned labels — enabling data-driven, segment-specific strategies.

**Type of Problem:** Unsupervised Learning — Clustering

---

## 2. Dataset Description

- **Name:** Mall Customer Segmentation Dataset
- **Original Source:** Kaggle — https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- **Local copy:** [`data/mall_customers.csv`](data/mall_customers.csv)
- **Samples:** 200 customers
- **Features:**

| Feature | Type | Description |
|---|---|---|
| CustomerID | Integer | Unique identifier |
| Gender | Categorical | Male / Female |
| Age | Integer | Age in years (18–70) |
| Annual_Income_kUSD | Integer | Annual income in thousands of USD |
| Spending_Score | Integer | Score 1–99 assigned by the mall based on spending behaviour |

- **Target:** None (unsupervised — no labels provided)

---

## 3. Data Preprocessing Steps

| Step | Action |
|---|---|
| Missing values | None found — no imputation required |
| Duplicates | None found |
| Feature encoding | Gender encoded (Female=1, Male=0) for compatibility |
| Feature scaling | `StandardScaler` applied to Age, Annual_Income_kUSD, Spending_Score |
| Train/test split | Not applicable for unsupervised clustering |

StandardScaler ensures no single feature dominates the Euclidean distance metric due to its scale. The fitted scalers are saved in `model/`.

---

## 4. Exploratory Data Analysis

EDA files are saved in [`results/`](results/):

| Plot | Description |
|---|---|
| `eda_gender_distribution.png` | Pie chart and count plot — 52.5% female, 47.5% male |
| `eda_feature_distributions.png` | Histograms + KDE for Age, Income, Spending Score |
| `eda_scatter_plots.png` | Age vs Spending and Income vs Spending, coloured by Gender |
| `eda_correlation_heatmap.png` | Correlation matrix — weak linear relationships |

**Key EDA findings:**
- No missing values or duplicates — data is clean.
- The income vs. spending scatter plot visually reveals approximately 5 natural customer clusters before any model is applied.
- Low correlation between features means PCA-based dimensionality reduction is not strictly necessary, but was used for 2D cluster visualisation.
- Spending Score shows near-uniform distribution, suggesting a diversity of customer behaviours.

---

## 5. Model(s) Used

Three unsupervised clustering algorithms were applied and compared:

1. **K-Means (k=5)** — partitions data by minimising within-cluster variance (WCSS)
2. **Hierarchical Clustering (Ward linkage, k=5)** — builds a cluster tree; no need to specify k upfront
3. **DBSCAN (eps=0.65, min_samples=5)** — density-based; discovers arbitrary shapes and flags outliers

---

## 6. Training Process

### K-Means
1. Run K-Means for k = 2 to 10, record inertia and Silhouette Score
2. Identify optimal k=5 from Elbow curve + Silhouette peak
3. Fit final `KMeans(n_clusters=5, n_init=20)` on scaled 3D features
4. Assign cluster labels; interpret centroids

### Hierarchical Clustering
1. Compute Ward linkage matrix on scaled features
2. Plot dendrogram; identify natural cut at Ward distance ≈ 6.5 → 5 clusters
3. Fit `AgglomerativeClustering(n_clusters=5, linkage='ward')`

### DBSCAN
1. Plot k-distance graph (k=5) to identify the "knee" at eps ≈ 0.65
2. Fit `DBSCAN(eps=0.65, min_samples=5)` on scaled features
3. Identify core clusters and noise points

All models and scalers saved to `model/` using `joblib`.

---

## 7. Evaluation Metrics and Results

Since no ground-truth labels are available, internal clustering metrics are used:

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Clusters |
|---|---|---|---|---|
| **K-Means (k=5)** | **0.4682** | **0.8564** | **198.79** | 5 |
| Hierarchical (Ward, k=5) | 0.4230 | 0.9178 | 171.07 | 5 |
| DBSCAN (eps=0.65, min=5) | 0.3553 | 0.7824 | 100.48 | 3 (+17 noise) |

**Metric interpretation:**
- Silhouette Score: how well each point fits its assigned cluster vs. neighbouring clusters (0.47 is a good value for real-world data)
- Davies-Bouldin: average cluster similarity — lower means more compact and separated clusters
- Calinski-Harabasz: ratio of between-cluster to within-cluster dispersion — K-Means scores highest

**K-Means Cluster Profiles:**

| Cluster | Age | Income | Spending | Interpretation |
|---|---|---|---|---|
| 0 | ~25 | Low (~26k) | High (~74) | Impulse Buyers |
| 1 | ~41 | Medium (~56k) | Medium (~49) | Standard Customers |
| 2 | ~28 | High (~85k) | High (~81) | **Target Customers** (most valuable) |
| 3 | ~50 | High (~89k) | Low (~17) | Careful Spenders |
| 4 | ~48 | Low (~29k) | Low (~21) | Budget Conscious |

Supporting plots: `results/kmeans_elbow_silhouette.png`, `results/kmeans_cluster_income_spending.png`, `results/kmeans_pca_projection.png`, `results/hierarchical_dendrogram.png`, `results/model_comparison.png`.

**Final model selected: K-Means (k=5)** — highest Silhouette and Calinski-Harabasz scores, clear cluster separation, and easily interpretable customer personas.

---

## 8. Answers to Report Questions

### Semantic vs Instance vs Panoptic Segmentation

These three approaches describe different ways of understanding an image at the pixel level in computer vision:

**Semantic Segmentation** labels every pixel with a class (e.g., car, road, sky), but all objects of the same class share one label — individual instances are not distinguished.

**Instance Segmentation** detects and separately outlines each individual object, even within the same class. Two people in a scene are labelled as two distinct masks. Background pixels are typically not classified.

**Panoptic Segmentation** combines both — it labels every pixel in the scene (like semantic segmentation) and also uniquely identifies each individual instance of countable objects (like instance segmentation), giving a complete scene understanding.

| | Semantic | Instance | Panoptic |
|---|---|---|---|
| Labels all pixels | ✓ | ✗ | ✓ |
| Distinguishes instances | ✗ | ✓ | ✓ |
| Background classified | ✓ | ✗ | ✓ |

---

### Fuzzy Logic vs Boolean Logic

**Boolean Logic** works with exactly two values: True (1) or False (0). Every proposition must be one or the other — there is no in-between. It suits deterministic, binary decisions.

**Fuzzy Logic** extends Boolean logic to allow any truth value in the continuous range [0.0, 1.0]. A concept like "tall" is not simply true or false — a person 5'10" might have a membership value of 0.7 in the "tall" set. This captures the gradation and uncertainty present in real-world reasoning.

| Aspect | Boolean Logic | Fuzzy Logic |
|---|---|---|
| Truth values | {0, 1} | [0.0, 1.0] |
| Handles uncertainty | No | Yes |
| Best for | On/off switches, exact decisions | Control systems, natural language |
| Example | Is age > 60? → True/False | Is person "old"? → 0.4 membership |

---

### Customer Dataset with Age, Annual Income, Spending Score — No Labels

**1. Is this supervised or unsupervised?**
This is an **unsupervised learning** problem. No target labels or output classes are given. The objective is to discover hidden groupings within the data using only the input features themselves.

**2. Which clustering algorithm to choose first and why?**
**K-Means** would be the first choice because:
- All three features are continuous and numeric, which is ideal for Euclidean distance-based clustering.
- The Elbow Method provides a principled, visual way to determine the number of clusters.
- Income vs Spending Score scatter plots of similar datasets consistently show well-separated, roughly spherical groups — the exact scenario where K-Means excels.
- It is computationally efficient and the cluster centroids are directly interpretable as customer personas.

**3. What assumptions does K-Means make?**
1. **Clusters are spherical** — uses Euclidean distance, so works best with round, compact clusters.
2. **Similar cluster sizes** — tends to produce clusters of roughly equal membership.
3. **Similar cluster variances** — struggles when clusters have very different spreads.
4. **k is known in advance** — the number of clusters must be specified before running.
5. **Features are on comparable scales** — requires StandardScaler preprocessing, otherwise high-range features dominate.

---

## 9. Conclusion

This project demonstrated a complete unsupervised machine learning workflow on retail customer data. Three clustering algorithms were applied and evaluated using internal metrics. K-Means with k=5 was the best-performing model (Silhouette = 0.468), producing five distinct, interpretable customer segments. Hierarchical Clustering independently confirmed the k=5 structure, while DBSCAN — though less suited to this dataset — successfully identified 17 borderline/outlier customers who do not fit neatly into any segment.

**Business impact:** The five customer segments can be used to design tailored marketing campaigns — for example, premium promotions for Cluster 2 (young, high-income, high-spending) and savings-focused offers for Cluster 4 (older, budget-conscious).

---

## References

1. Choudhary, V. (2019). *Customer Segmentation Tutorial in Python*. Kaggle. https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
3. Zadeh, L.A. (1965). *Fuzzy sets*. Information and Control, 8(3), 338–353.
4. Ester, M. et al. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD-96.

---

## Repository Structure

```
.
├── data/
│   └── mall_customers.csv              # 200-customer dataset
├── model/
│   ├── kmeans_k5.pkl                   # Trained K-Means model
│   ├── hierarchical_k5.pkl             # Trained Hierarchical model
│   ├── dbscan.pkl                      # Trained DBSCAN model
│   ├── scaler_3d.pkl                   # StandardScaler (3 features)
│   └── scaler_2d.pkl                   # StandardScaler (2 features)
├── notebook/
│   ├── Minor_Project_2_Customer_Segmentation.ipynb
│   └── pipeline_script.py
├── results/
│   ├── eda_gender_distribution.png
│   ├── eda_feature_distributions.png
│   ├── eda_scatter_plots.png
│   ├── eda_correlation_heatmap.png
│   ├── kmeans_elbow_silhouette.png
│   ├── kmeans_cluster_income_spending.png
│   ├── kmeans_pca_projection.png
│   ├── hierarchical_dendrogram.png
│   ├── hierarchical_cluster_income_spending.png
│   ├── dbscan_kdistance.png
│   ├── dbscan_cluster_income_spending.png
│   ├── model_comparison.png
│   ├── kmeans_cluster_profiles.csv
│   ├── metrics_summary.json
│   └── metrics_summary.txt
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

# Run as a script
python notebook/pipeline_script.py

# Or open the notebook
jupyter notebook notebook/Minor_Project_2_Customer_Segmentation.ipynb
```

## Load a Saved Model

```python
import joblib, numpy as np

km = joblib.load("model/kmeans_k5.pkl")
scaler = joblib.load("model/scaler_3d.pkl")

# X_new: shape (n, 3) → [Age, Annual_Income_kUSD, Spending_Score]
X_scaled = scaler.transform(X_new)
cluster_labels = km.predict(X_scaled)
```
