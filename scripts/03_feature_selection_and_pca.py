"""
================================================================================
 Script Name   : 03_feature_selection_and_pca.py
 Location      : scripts/03_feature_selection_and_pca.py
 Author        : Fady Romany
 Date          : 2025-09-13
 Requirements  : pandas, numpy, scikit-learn, matplotlib, seaborn
 Input         : data/heart_disease_cleaned.csv
 Output        : results/feature_importances.csv, results/pca_variance.png,
                 results/rfe_selected_features.csv
 Description   :
    - Performs feature selection using RandomForest importance scores.
    - Saves feature rankings for downstream analysis.
    - Runs Principal Component Analysis (PCA) to reduce dimensionality.
    - Visualizes explained variance ratio to guide component selection.
    - Applies Recursive Feature Elimination (RFE) to select top predictors.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# File paths
# ----------------------------------------------------------------------
DATA = "data/heart_disease_cleaned.csv"
os.makedirs("results", exist_ok=True)


def main():
    """
    Perform feature selection and dimensionality reduction.
    
    Workflow:
        1. Load cleaned dataset.
        2. Standardize features.
        3. Run PCA and plot explained variance.
        4. Train RandomForest to estimate feature importances.
        5. Apply Recursive Feature Elimination (RFE).
        6. Save results (plots + CSVs).
    
    Returns:
        None. Saves outputs to results/ folder.
    """
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=['target']), df['target']

    # ------------------------------------------------------------------
    # PCA for dimensionality reduction
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(Xs)

    evr = pca.explained_variance_ratio_
    cumvar = evr.cumsum()

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(evr) + 1), cumvar, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.grid(True)
    plt.savefig("results/pca_variance.png", dpi=150)
    print("✅ Saved PCA variance plot to results/pca_variance.png")

    # ------------------------------------------------------------------
    # Feature importance via RandomForest
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv("results/feature_importances.csv")
    print("✅ Saved feature importances to results/feature_importances.csv")

    # ------------------------------------------------------------------
    # Recursive Feature Elimination (RFE)
    # ------------------------------------------------------------------
    rfe = RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        n_features_to_select=min(8, X.shape[1]),
        step=1
    )
    rfe.fit(X, y)

    selected = pd.Series(rfe.support_, index=X.columns)
    selected.to_csv("results/rfe_selected_features.csv")
    print("✅ Saved RFE selected features to results/rfe_selected_features.csv")


if __name__ == "__main__":
    main()
