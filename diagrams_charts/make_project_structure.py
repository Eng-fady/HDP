from graphviz import Digraph

dot = Digraph(comment="Project Structure", format="png")
dot.attr(size="8", rankdir="LR")

dot.node("root", "Heart_Disease_Project/")

# Main folders
for folder in ["data/", "models/", "notebooks/", "results/", "scripts/", "ui/"]:
    dot.node(folder, folder, shape="folder")
    dot.edge("root", folder)

# Data
dot.edge("data/", "heart_disease.csv")
dot.edge("data/", "heart_disease_cleaned.csv")
dot.edge("data/", "test_patients.csv")

# Models
dot.edge("models/", "final_model.pkl")

# Scripts
scripts = [
    "01_data_preprocessing.py",
    "02_train_baseline_models.py",
    "05_hyperparameter_tuning.py",
    "bulk_predict.py",
]
for s in scripts:
    dot.edge("scripts/", s)

# UI
dot.edge("ui/", "app.py")

dot.render("project_structure", cleanup=True)
print("✅ Saved project_structure.png")