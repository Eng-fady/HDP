# scripts/01_data_preprocessing.py
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer

DATA_IN = "data/heart_disease.csv"
DATA_OUT = "data/heart_disease_cleaned.csv"

def load_raw(path):
    df = pd.read_csv(path, na_values=['?', 'NA', 'NaN', ''])
    return df

def detect_columns(df, target_col='target'):
    cols = df.columns.tolist()
    if target_col not in cols:
        for alt in ['diagnosis', 'target', 'heartdisease', 'num']:
            if alt in cols:
                target_col = alt
                break
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    numeric = [c for c in numeric if c != target_col]
    possible_cat = [c for c in df.columns if c not in numeric and c != target_col]
    for c in numeric:
        if df[c].nunique() <= 10:
            possible_cat.append(c)
    numeric = [c for c in df.columns if c not in possible_cat and c != target_col]
    return numeric, possible_cat, target_col

def clean(df, numeric, categorical, target_col):
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'ca' in df.columns:
        df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    if 'thal' in df.columns:
        df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

    num_imp = SimpleImputer(strategy='median')
    cat_imp = SimpleImputer(strategy='most_frequent')

    df_num = pd.DataFrame(num_imp.fit_transform(df[numeric]), columns=numeric)
    if len(categorical) > 0:
        df_cat = pd.DataFrame(cat_imp.fit_transform(df[categorical]), columns=categorical)
        df_clean = pd.concat([df_num, df_cat, df[[target_col]].reset_index(drop=True)], axis=1)
    else:
        df_clean = pd.concat([df_num, df[[target_col]].reset_index(drop=True)], axis=1)

    if df_clean[target_col].nunique() > 2:
        df_clean[target_col] = (df_clean[target_col] > 0).astype(int)

    return df_clean

def main():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(DATA_IN):
        raise FileNotFoundError(f"Place the raw CSV at {DATA_IN}")
    df = load_raw(DATA_IN)
    numeric, categorical, target_col = detect_columns(df)
    print("Numeric cols:", numeric)
    print("Categorical cols:", categorical)
    df_clean = clean(df, numeric, categorical, target_col)
    df_clean.to_csv(DATA_OUT, index=False)
    print(f"Saved cleaned dataset to {DATA_OUT}. Shape: {df_clean.shape}")

if __name__ == "__main__":
    main()
