# Heart_Disease_Project

This repository contains a reproducible ML pipeline for the Heart Disease UCI dataset.

Folder structure:
- data/: raw and cleaned dataset (place `heart_disease.csv` here)
- notebooks/: EDA, PCA, feature selection, modeling notebooks
- scripts/: reproducible scripts to run preprocessing, training, PCA, clustering, tuning
- models/: saved pipeline / model artifacts (.pkl)
- ui/: Streamlit application (app.py)
- results/: plots and evaluation metrics
- deployment/: ngrok / deployment instructions

Quick start (after placing `heart_disease.csv` inside `data/` and activating your venv):

```powershell
python scripts/01_data_preprocessing.py
python scripts/02_train_baseline_models.py
python scripts/03_feature_selection_and_pca.py
python scripts/04_clustering.py
python scripts/05_hyperparameter_tuning.py  # optional (long)
streamlit run ui/app.py
```

See `deployment/ngrok_setup.txt` for a short ngrok guide.

This skeleton was created to match the project brief provided in your PDF while adding a `scripts/` folder for reproducible command-line execution.
