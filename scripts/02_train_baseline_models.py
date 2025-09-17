"""
================================================================================
 Script Name   : 02_train_baseline_models.py
 Location      : scripts/02_train_baseline_models.py
 Author        : Fady Romany
 Date          : 2025-09-13
 Requirements  : pandas, numpy, scikit-learn, joblib
 Input         : data/heart_disease_cleaned.csv
 Output        : models/final_model.pkl, results/model_comparison.csv,
                 results/evaluation_reports/
 Description   :
    - Trains multiple baseline models (Logistic, Decision Tree, Random Forest, SVM).
    - Preprocessing: imputation, scaling, one-hot encoding.
    - Evaluates models using accuracy, precision, recall, F1, and ROC-AUC.
    - Selects and saves best-performing model pipeline and metadata.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score
)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
DATA = "data/heart_disease_cleaned.csv"
MODEL_OUT = "models/final_model.pkl"

# Ensure output directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('results/evaluation_reports', exist_ok=True)


def load_data(path=DATA):
    """Load cleaned dataset and ensure target is binary."""
    df = pd.read_csv(path)
    if 'target' not in df.columns:
        raise ValueError("No target column found in dataset.")
    if df['target'].nunique() > 2:
        df['target'] = (df['target'] > 0).astype(int)
    return df


def detect_cols(df, target='target'):
    """Identify numeric and categorical feature columns."""
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    numeric = [c for c in numeric if c != target]
    categorical = [c for c in df.columns if c not in numeric + [target]]
    return numeric, categorical


def build_preprocessor(numeric_features, categorical_features):
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


def evaluate_and_save(pipeline, X_test, y_test, name):
    """
    Evaluate trained model on test set and save report.

    Returns:
        dict: Metrics including accuracy, precision, recall, f1, and ROC AUC.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['clf'], "predict_proba") else None

    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': auc
    }

    with open(f"results/evaluation_reports/{name}_report.txt", "w") as f:
        f.write(f"Model: {name}\n")
        f.write(classification_report(y_test, y_pred))
        if auc is not None:
            f.write(f"\nROC AUC: {auc:.4f}\n")

    return metrics


def main():
    """Train baseline models, evaluate them, and save best pipeline."""
    df = load_data()
    X, y = df.drop(columns=['target']), df['target']

    numeric_features, categorical_features = detect_cols(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Candidate models
    models = {
        'logistic': LogisticRegression(max_iter=500, solver='liblinear'),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'svm': SVC(probability=True, gamma='scale', random_state=42)
    }

    results, best_model_name, best_pipeline, best_score = [], None, None, -np.inf

    # Train and evaluate each model
    for name, clf in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_and_save(pipe, X_test, y_test, name)
        results.append(metrics)

        print(f"[{name}] accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']}")
        score = metrics['roc_auc'] if metrics['roc_auc'] is not None else metrics['f1']

        if score > best_score:
            best_score, best_model_name, best_pipeline = score, name, pipe

    # Save model comparison
    pd.DataFrame(results).sort_values('roc_auc', ascending=False).to_csv("results/model_comparison.csv", index=False)
    print("✅ Saved model comparison to results/model_comparison.csv")

    # Save best pipeline with metadata
    metadata = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'categorical_options': {c: list(df[c].dropna().unique()) for c in categorical_features},
        'feature_order': X.columns.tolist()
    }
    joblib.dump({'pipeline': best_pipeline, 'metadata': metadata}, MODEL_OUT)
    print(f"✅ Saved best pipeline ({best_model_name}) to {MODEL_OUT}")


if __name__ == "__main__":
    main()
