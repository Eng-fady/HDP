# scripts/03_feature_selection_and_pca.py
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA = "data/heart_disease_cleaned.csv"
os.makedirs('results', exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=['target'])
    y = df['target']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(Xs)
    evr = pca.explained_variance_ratio_
    cumvar = evr.cumsum()
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(1, len(evr)+1), cumvar, marker='o')
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.grid(True)
    plt.savefig("results/pca_variance.png", dpi=150)
    print("Saved PCA variance plot to results/pca_variance.png")

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv("results/feature_importances.csv")
    print("Saved feature importances to results/feature_importances.csv")

    rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=min(8, X.shape[1]), step=1)
    rfe.fit(X, y)
    selected = pd.Series(rfe.support_, index=X.columns)
    selected.to_csv("results/rfe_selected_features.csv")
    print("Saved RFE selection to results/rfe_selected_features.csv")

if __name__ == "__main__":
    main()
