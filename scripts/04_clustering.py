"""
================================================================================
 Script Name   : 04_clustering.py
 Location      : scripts/04_clustering.py
 Author        : Fady Romany
 Date          : 2025-09-13
 Requirements  : pandas, numpy, matplotlib, scikit-learn, scipy
 Input         : data/heart_disease_cleaned.csv
 Output        : results/kmeans_elbow.png, results/dendrogram_truncated.png,
                 results/cluster_vs_true.csv
 Description   :
    - Applies unsupervised learning (KMeans & hierarchical clustering).
    - Generates elbow plot to guide optimal k selection.
    - Evaluates cluster quality using silhouette score.
    - Creates truncated dendrogram visualization.
    - Compares clustering assignments to ground truth labels.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# ----------------------------------------------------------------------
# File paths
# ----------------------------------------------------------------------
DATA = "data/heart_disease_cleaned.csv"
os.makedirs("results", exist_ok=True)


def main():
    """
    Run clustering analysis using KMeans and hierarchical clustering.

    Workflow:
        1. Load cleaned dataset.
        2. Standardize features.
        3. Run KMeans for range of k, save elbow plot.
        4. Compute silhouette scores for quick validation.
        5. Generate truncated dendrogram via hierarchical clustering.
        6. Compare cluster assignments vs true labels.
    
    Returns:
        None. Saves all results under results/ folder.
    """
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=['target']), df['target']

    # ------------------------------------------------------------------
    # Standardize features (important for distance-based clustering)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # KMeans Elbow Method
    # ------------------------------------------------------------------
    inertias = []
    K = range(2, 8)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("KMeans Elbow Plot")
    plt.grid(True)
    plt.savefig("results/kmeans_elbow.png", dpi=150)
    print("✅ Saved KMeans elbow plot to results/kmeans_elbow.png")

    # ------------------------------------------------------------------
    # Silhouette scores (quick validation of cluster quality)
    # ------------------------------------------------------------------
    for k in [2, 3]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        print(f"Silhouette score for k={k}: {sil:.4f}")

    # ------------------------------------------------------------------
    # Hierarchical clustering (truncated dendrogram for clarity)
    # ------------------------------------------------------------------
    Z = linkage(Xs[:200], method="ward")  # subset to 200 samples for readability
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram (truncated)")
    plt.savefig("results/dendrogram_truncated.png", dpi=150)
    print("✅ Saved dendrogram to results/dendrogram_truncated.png")

    # ------------------------------------------------------------------
    # Compare clustering vs true labels
    # ------------------------------------------------------------------
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    clabels = km.fit_predict(Xs)

    df_comp = pd.DataFrame({"true": y, "cluster": clabels})
    cross = pd.crosstab(df_comp["cluster"], df_comp["true"])
    cross.to_csv("results/cluster_vs_true.csv")
    print("✅ Saved cluster vs true cross-tab to results/cluster_vs_true.csv")


if __name__ == "__main__":
    main()
