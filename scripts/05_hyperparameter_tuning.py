"""
================================================================================
 Script Name   : 05_hyperparameter_tuning.py
 Location      : scripts/05_hyperparameter_tuning.py
 Author        : Fady Romany
 Date          : 2025-09-13
 Requirements  : pandas, numpy, scikit-learn, joblib
 Input         : data/heart_disease_cleaned.csv
 Output        : models/final_model.pkl
 Description   :
    - Performs hyperparameter tuning for RandomForest and SVM classifiers.
    - Uses GridSearchCV with StratifiedKFold cross-validation.
    - Evaluates tuned models on ROC AUC.
    - Saves the best-performing pipeline and metadata for later inference.
================================================================================
"""

import joblib
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------
# Constants and paths
# ----------------------------------------------------------------------
DATA = "data/heart_disease_cleaned.csv"
MODEL_OUT = "models/final_model.pkl"

# Ensure output directories exist
os.makedirs('models', exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_data():
    """
    Load cleaned dataset and separate features/target.
    
    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target labels
    """
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=['target']), df['target']
    return X, y


def detect_cols(df):
    """
    Detect numeric and categorical features automatically.

    Args:
        df (pd.DataFrame): Full dataset including target.
    Returns:
        numeric (list): Numeric feature columns
        categorical (list): Categorical feature columns
    """
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    numeric = [c for c in numeric if c != 'target']
    categorical = [c for c in df.columns if c not in numeric + ['target']]
    return numeric, categorical


def build_preprocessor(numeric_features, categorical_features):
    """
    Build preprocessing pipeline for numeric and categorical columns.

    Numeric: median imputation + scaling.
    Categorical: most-frequent imputation + one-hot encoding.
    
    Args:
        numeric_features (list): Numeric columns
        categorical_features (list): Categorical columns
    Returns:
        ColumnTransformer: Preprocessor object
    """
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ])


def main():
    """Main execution: hyperparameter tuning and model selection."""
    # Load data and preprocess column groups
    X, y = load_data()
    numeric, categorical = detect_cols(pd.concat([X, y], axis=1))
    pre = build_preprocessor(numeric, categorical)

    # Train/test split with stratification to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # Random Forest: define pipeline and hyperparameter grid
    # ------------------------------------------------------------------
    rf_pipe = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    rf_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 6, 12],
        'clf__min_samples_split': [2, 5]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs_rf = GridSearchCV(
        rf_pipe,
        rf_param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    gs_rf.fit(X_train, y_train)
    print("Best RF params:", gs_rf.best_params_)

    rf_best = gs_rf.best_estimator_
    rf_auc = roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1])
    print(f"RF test ROC AUC: {rf_auc:.4f}")

    # ------------------------------------------------------------------
    # Support Vector Classifier: pipeline and grid
    # ------------------------------------------------------------------
    svc_pipe = Pipeline([
        ('pre', pre),
        ('clf', SVC(probability=True, random_state=42))
    ])

    svc_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 'auto'],
        'clf__kernel': ['rbf']
    }

    gs_svc = GridSearchCV(
        svc_pipe,
        svc_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    gs_svc.fit(X_train, y_train)

    svc_best = gs_svc.best_estimator_
    svc_auc = roc_auc_score(y_test, svc_best.predict_proba(X_test)[:, 1])
    print(f"SVC test ROC AUC: {svc_auc:.4f}")

    # ------------------------------------------------------------------
    # Model selection: keep best-performing model
    # ------------------------------------------------------------------
    final_model = rf_best if rf_auc >= svc_auc else svc_best

    # Metadata useful for inference and UI
    metadata = {
        'numeric_features': numeric,
        'categorical_features': categorical,
        'categorical_options': {c: list(X[c].dropna().unique()) for c in categorical},
        'feature_order': X.columns.tolist()
    }

    # Save model + metadata together
    joblib.dump({'pipeline': final_model, 'metadata': metadata}, MODEL_OUT)
    print(f"✅ Saved tuned model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
