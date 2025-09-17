"""
================================================================================
 Script Name   : 01_data_preprocessing.py
 Location      : scripts/01_data_preprocessing.py
 Author        : Fady Romany
 Date          : 2025-09-13
 Requirements  : pandas, numpy, scikit-learn
 Input         : data/heart_disease.csv (raw dataset)
 Output        : data/heart_disease_cleaned.csv (cleaned dataset)
 Description   :
    - Loads raw UCI Heart Disease dataset.
    - Detects numeric and categorical columns automatically.
    - Handles missing values via median/mode imputation.
    - Ensures binary target column for classification.
    - Saves the cleaned dataset for downstream modeling.
================================================================================
"""

import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------
# File paths
# ----------------------------------------------------------------------
DATA_IN = "data/heart_disease.csv"
DATA_OUT = "data/heart_disease_cleaned.csv"


def load_raw(path):
    """
    Load raw dataset and normalize missing value markers.
    
    Args:
        path (str): Path to raw dataset.
    Returns:
        pd.DataFrame: Raw dataframe with NaNs handled.
    """
    return pd.read_csv(path, na_values=['?', 'NA', 'NaN', ''])


def detect_columns(df, target_col='target'):
    """
    Detect numeric and categorical columns automatically.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of target column, defaults to 'target'.

    Returns:
        numeric (list): Numeric columns.
        categorical (list): Categorical columns.
        target_col (str): Confirmed target column name.
    """
    cols = df.columns.tolist()

    # Handle alternative target naming
    if target_col not in cols:
        for alt in ['diagnosis', 'target', 'heartdisease', 'num']:
            if alt in cols:
                target_col = alt
                break

    # Detect numeric columns
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    numeric = [c for c in numeric if c != target_col]

    # Identify categorical candidates (low-cardinality numeric variables)
    possible_cat = [c for c in df.columns if c not in numeric and c != target_col]
    for c in numeric:
        if df[c].nunique() <= 10:
            possible_cat.append(c)

    numeric = [c for c in df.columns if c not in possible_cat and c != target_col]
    return numeric, possible_cat, target_col


def clean(df, numeric, categorical, target_col):
    """
    Clean dataset: impute missing values, convert dtypes, and ensure binary target.

    Args:
        df (pd.DataFrame): Input dataframe.
        numeric (list): Numeric features.
        categorical (list): Categorical features.
        target_col (str): Target column name.

    Returns:
        pd.DataFrame: Cleaned dataset ready for modeling.
    """
    # Force numeric conversion for problematic columns
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'ca' in df.columns:
        df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    if 'thal' in df.columns:
        df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

    # Imputation
    num_imp = SimpleImputer(strategy='median')
    cat_imp = SimpleImputer(strategy='most_frequent')

    df_num = pd.DataFrame(num_imp.fit_transform(df[numeric]), columns=numeric)
    if len(categorical) > 0:
        df_cat = pd.DataFrame(cat_imp.fit_transform(df[categorical]), columns=categorical)
        df_clean = pd.concat([df_num, df_cat, df[[target_col]].reset_index(drop=True)], axis=1)
    else:
        df_clean = pd.concat([df_num, df[[target_col]].reset_index(drop=True)], axis=1)

    # Ensure binary classification target
    if df_clean[target_col].nunique() > 2:
        df_clean[target_col] = (df_clean[target_col] > 0).astype(int)

    return df_clean


def main():
    """Main execution: preprocess raw dataset and save cleaned version."""
    os.makedirs('data', exist_ok=True)

    if not os.path.exists(DATA_IN):
        raise FileNotFoundError(f"Place the raw CSV at {DATA_IN}")

    df = load_raw(DATA_IN)
    numeric, categorical, target_col = detect_columns(df)
    print("Numeric cols:", numeric)
    print("Categorical cols:", categorical)

    df_clean = clean(df, numeric, categorical, target_col)
    df_clean.to_csv(DATA_OUT, index=False)
    print(f"✅ Saved cleaned dataset to {DATA_OUT}. Shape: {df_clean.shape}")


if __name__ == "__main__":
    main()
