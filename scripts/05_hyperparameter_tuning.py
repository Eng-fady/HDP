# scripts/05_hyperparameter_tuning.py
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

DATA = "data/heart_disease_cleaned.csv"
MODEL_OUT = "models/final_model.pkl"
os.makedirs('models', exist_ok=True)

def load_data():
    df = pd.read_csv(DATA)
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def detect_cols(df):
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    numeric = [c for c in numeric if c != 'target']
    categorical = [c for c in df.columns if c not in numeric + ['target']]
    return numeric, categorical

def build_preprocessor(numeric_features, categorical_features):
    from sklearn.pipeline import Pipeline
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    from sklearn.compose import ColumnTransformer
    return ColumnTransformer([('num', num_pipe, numeric_features), ('cat', cat_pipe, categorical_features)])

def main():
    X, y = load_data()
    numeric, categorical = detect_cols(pd.concat([X, y], axis=1))
    pre = build_preprocessor(numeric, categorical)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    rf_pipe = Pipeline([('pre', pre), ('clf', RandomForestClassifier(random_state=42))])
    rf_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 6, 12],
        'clf__min_samples_split': [2, 5]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs_rf = GridSearchCV(rf_pipe, rf_param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    gs_rf.fit(X_train, y_train)
    print("Best RF params:", gs_rf.best_params_)
    rf_best = gs_rf.best_estimator_
    rf_auc = roc_auc_score(y_test, rf_best.predict_proba(X_test)[:,1])
    print(f"RF test ROC AUC: {rf_auc:.4f}")

    svc_pipe = Pipeline([('pre', pre), ('clf', SVC(probability=True, random_state=42))])
    svc_dist = {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto'], 'clf__kernel': ['rbf']}
    gs_svc = GridSearchCV(svc_pipe, svc_dist, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    gs_svc.fit(X_train, y_train)
    svc_best = gs_svc.best_estimator_
    svc_auc = roc_auc_score(y_test, svc_best.predict_proba(X_test)[:,1])
    print(f"SVC test ROC AUC: {svc_auc:.4f}")

    final_model = rf_best if rf_auc >= svc_auc else svc_best
    metadata = {
        'numeric_features': numeric,
        'categorical_features': categorical,
        'categorical_options': {c: list(X[c].dropna().unique()) for c in categorical},
        'feature_order': X.columns.tolist()
    }
    joblib.dump({'pipeline': final_model, 'metadata': metadata}, MODEL_OUT)
    print(f"Saved tuned model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
