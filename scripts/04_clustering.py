# scripts/04_clustering.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

DATA = "data/heart_disease_cleaned.csv"
os.makedirs('results', exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=['target'])
    y = df['target']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    inertias = []
    K = range(2, 8)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(K, inertias, marker='o')
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.title('KMeans elbow')
    plt.grid(True)
    plt.savefig('results/kmeans_elbow.png', dpi=150)

    for k in [2,3]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        print(f"KMeans k={k} silhouette={sil:.4f}")

    Z = linkage(Xs[:200], method='ward')
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.savefig('results/dendrogram_truncated.png', dpi=150)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    clabels = km.fit_predict(Xs)
    df_comp = pd.DataFrame({'true': y, 'cluster': clabels})
    cross = pd.crosstab(df_comp['cluster'], df_comp['true'])
    cross.to_csv('results/cluster_vs_true.csv')
    print("Saved cluster vs true cross-tab to results/cluster_vs_true.csv")

if __name__ == "__main__":
    main()
