from graphviz import Digraph

dot = Digraph(comment="Project Structure", format="svg")
dot.attr(size="8", rankdir="LR", dpi ="300")

dot.node("root", "Heart_Disease_Project/")

# Main folders
for folder in ["data/", "models/", "notebooks/", "results/", "scripts/", "ui/"]:
    dot.node(folder, folder, shape="folder")
    dot.edge("root", folder)

# Data
dot.node("heart_disease.csv", "heart_disease.csv", shape="note")
dot.node("heart_disease_cleaned.csv", "heart_disease_cleaned.csv", shape="note")
dot.node("test_patients.csv", "test_patients.csv", shape="note")
dot.edge("data/", "heart_disease.csv")
dot.edge("data/", "heart_disease_cleaned.csv")
dot.edge("data/", "test_patients.csv")

# Models
dot.node("final_model.pkl", "final_model.pkl", shape="note")
dot.edge("models/", "final_model.pkl")

# Scripts
scripts = [
    "01_data_preprocessing.py",
    "02_train_baseline_models.py",
    "03_feature_selection_and_pca.py",
    "04_clustering.py",
    "05_hyperparameter_tuning.py",
    "bulk_predict.py",
]
for s in scripts:
    dot.node(s, s, shape="note")
    dot.edge("scripts/", s)

# UI
dot.node("app.py", "app.py", shape="note")
dot.edge("ui/", "app.py")

dot.render("project_structure", format="svg", cleanup=True)
print("✅ Saved project_structure.svg")
