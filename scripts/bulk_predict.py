"""
================================================================================
 Script Name   : bulk_predict.py
 Author        : Fady Romany
 Location      : scripts/bulk_predict.py
 Date          : 2025-09-13
 Requirements  : pandas, joblib
 Input         : data/test_patients.csv, models/final_model.pkl
 Output        : results/test_predictions.csv
 Description   :
    - Loads saved pipeline (with metadata).
    - Reads patient data from CSV.
    - Produces predictions (binary + risk label).
    - Saves predictions to results directory.
================================================================================
"""

import pandas as pd
import joblib
import os

MODEL_PATH = "models/final_model.pkl"
TEST_FILE = "data/test_patients.csv"
OUTPUT_FILE = "results/test_predictions.csv"

os.makedirs("results", exist_ok=True)


def main():
    """Run bulk prediction on test CSV."""
    loaded = joblib.load(MODEL_PATH)
    pipeline, meta = loaded["pipeline"], loaded["metadata"]

    df = pd.read_csv(TEST_FILE)

    preds = pipeline.predict(df)
    labels = ["Low Risk" if p == 0 else "High Risk" for p in preds]

    df["prediction"] = labels
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
