import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("models", "final_model.pkl")

# Load trained model dictionary
model_dict = joblib.load(MODEL_PATH)
model = model_dict["pipeline"]  # the actual sklearn pipeline

# Load test patients
df = pd.read_csv("data/test_patients.csv")

# Make predictions
predictions = model.predict(df)

# Add predictions to the dataframe
df["prediction"] = predictions
df["prediction"] = df["prediction"].map({0: "Low Risk", 1: "High Risk"})

# Save results
df.to_csv("results/test_predictions.csv", index=False)

print("✅ Predictions saved to results/test_predictions.csv")
print(df)
