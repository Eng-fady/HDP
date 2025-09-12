import pandas as pd

col_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","target"
]

# Now point to your project data folder
df = pd.read_csv("data/processed.cleveland.data", names=col_names)

df.to_csv("data/heart_disease.csv", index=False)

print("✅ Saved cleaned dataset to data/heart_disease.csv with headers.")
