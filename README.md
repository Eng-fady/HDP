It seems like I can’t do more advanced data analysis right now. Please try again later.

That said, I already have the full, polished `README.md` content ready for you. Since I can’t generate the file directly at the moment, here’s the **final version you can copy-paste into your `README.md`** file:

---

````markdown
# 🫀 Heart Disease Risk Prediction

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](#run-the-app)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?logo=scikit-learn)](https://scikit-learn.org/)  
[![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)](https://www.python.org/)

---

## TL;DR — run the app (fast)
```powershell
# from project root (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/01_data_preprocessing.py
python scripts/02_train_baseline_models.py
streamlit run ui/app.py
````

---

# Table of contents

* [Overview](#overview)
* [Prerequisites & Quick Setup (Windows)](#prerequisites--quick-setup-windows)
* [Dataset — exact steps to prepare `data/heart_disease.csv`](#dataset---exact-steps-to-prepare-dataheart_diseasecsv)
* [Run the pipeline (detailed)](#run-the-pipeline-detailed)
* [Streamlit app — manual + CSV upload](#streamlit-app---manual--csv-upload)
* [CSV sample](#csv-sample-place-at-data-test_patientscsv)
* [Diagrams](#diagrams---generate-and-include-them-in-readme)
* [Troubleshooting](#troubleshooting-common-problems)
* [Git & repo hygiene tips](#git--repo-hygiene-tips)
* [Disclaimer & author](#final-notes--legal)

---

# Overview

This repository contains a reproducible pipeline for predicting heart disease (UCI dataset).
It includes: preprocessing, feature selection/PCA, baseline model training, clustering, hyperparameter tuning, model export, a Streamlit UI (single-entry + CSV upload), and scripts for bulk predictions.

---

# Prerequisites & Quick Setup (Windows)

1. Install **Python 3.11** (64-bit) and add to PATH.
2. Open **PowerShell** and run:

```powershell
cd C:\Users\PC\Desktop\Heart_Disease_Project
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify:

```powershell
python --version
pip --version
dot -V
```

---

# Dataset — exact steps to prepare `data/heart_disease.csv`

1. Extract `heart+disease.zip`.
2. Copy `processed.cleveland.data` into `data/`.
3. Convert with the script:

```python
# scripts/convert_to_csv.py
import pandas as pd
col_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","target"
]
df = pd.read_csv("data/processed.cleveland.data", names=col_names)
df.to_csv("data/heart_disease.csv", index=False)
print("Saved data/heart_disease.csv")
```

Run:

```powershell
python scripts/convert_to_csv.py
```

---

# Run the pipeline (detailed)

**1. Preprocess**

```powershell
python scripts/01_data_preprocessing.py
```

Creates `data/heart_disease_cleaned.csv`.

**2. Train baseline models**

```powershell
python scripts/02_train_baseline_models.py
```

Creates `models/final_model.pkl` and results in `results/`.

**3. PCA & feature selection (optional)**

```powershell
python scripts/03_feature_selection_and_pca.py
```

**4. Clustering (optional)**

```powershell
python scripts/04_clustering.py
```

**5. Hyperparameter tuning (optional, long)**

```powershell
python scripts/05_hyperparameter_tuning.py
```

**6. Bulk predict**

```powershell
python scripts/bulk_predict.py
```

Outputs to `results/test_predictions.csv`.

---

# Streamlit app — manual + CSV upload

```powershell
streamlit run ui/app.py
```

* Manual Entry: enter patient data one by one.
* Upload CSV: upload file with same columns as training data, returns predictions in table and downloadable CSV.

---

# CSV sample (place at `data/test_patients.csv`)

```csv
age,trestbps,chol,thalach,oldpeak,sex,cp,fbs,restecg,exang,slope,ca,thal
54,130,250,160,1.0,1,0,0,1,0,2,0,2
35,120,200,180,0.0,0,2,0,0,0,0,0,1
70,145,300,130,3.5,1,0,1,1,1,2,3,3
```

---

# Diagrams — generate and include them in README

Scripts generate three diagrams:

* `project_structure.png`
* `workflow_diagram.png`
* `system_architecture.png`

Run with Graphviz installed:

```powershell
python make_project_structure.py
python make_workflow_diagram.py
python make_system_architecture.py
```

Commit PNGs to render in GitHub.

---

# Troubleshooting (common problems)

**Activate.ps1 blocked**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**OneHotEncoder error**
Use `sparse_output=False` instead of `sparse=False`.

**Graphviz not found**
Add `C:\Program Files\Graphviz\bin` to PATH. Verify with `dot -V`.

**Model not found**
Ensure `ui/app.py` uses:

```python
MODEL_PATH = os.path.join('models','final_model.pkl')
```

**dict has no attribute predict**
Access pipeline with:

```python
loaded = joblib.load("models/final_model.pkl")
pipeline = loaded['pipeline']
```

---

# Git & repo hygiene tips

```
venv/
__pycache__/
*.pyc
.DS_Store
models/
results/
data/processed.*
```

---

# Final notes & legal

This repository is for educational purposes only, not for medical use.

---

## Author

Developed by [Fady Romany](https://github.com/Eng-fady)
🎯 For learning, experimentation, and educational demonstrations.

```