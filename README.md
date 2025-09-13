## 📄 `README.md`

```markdown
# 🫀 Heart Disease Risk Prediction

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](#run-the-app)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?logo=scikit-learn)](https://scikit-learn.org/)  
[![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)](https://www.python.org/)  

## 📌 Overview
This project demonstrates a **machine learning pipeline** to predict the likelihood of heart disease using the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease).  

The system supports:
- ✅ **Single-patient prediction** via a **Streamlit UI form**  
- ✅ **Bulk predictions** via **CSV upload**  
- ✅ End-to-end ML workflow: preprocessing → training → evaluation → deployment  

⚠️ **Disclaimer:** This project is for **educational purposes only** and should not be used for real medical decisions.

---

## 📂 Project Structure
```

Heart\_Disease\_Project/
│── data/                # raw and processed datasets
│   ├── heart\_disease.csv
│   ├── heart\_disease\_cleaned.csv
│   └── test\_patients.csv
│── models/              # trained model artifacts (.pkl)
│   └── final\_model.pkl
│── notebooks/           # Jupyter notebooks for EDA/experiments
│── results/             # evaluation reports, predictions
│── scripts/             # preprocessing, training, evaluation
│   ├── 01\_data\_preprocessing.py
│   ├── 02\_train\_baseline\_models.py
│   ├── 05\_hyperparameter\_tuning.py
│   ├── bulk\_predict.py
│   └── ...
│── ui/                  # Streamlit web application
│   └── app.py
│── requirements.txt     # Python dependencies
│── README.md            # project documentation
│── .gitignore

````

---

## ⚙️ Installation

Clone the repo and set up a Python environment (Python 3.11 recommended):

```bash
git clone https://github.com/YourUsername/Heart_Disease_Project.git
cd Heart_Disease_Project

# Create virtual environment
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🧪 Run the Pipeline

### 1. Data Preprocessing

```bash
python scripts/01_data_preprocessing.py
```

* Cleans raw data from `data/heart_disease.csv`
* Outputs: `data/heart_disease_cleaned.csv`

### 2. Train Baseline Models

```bash
python scripts/02_train_baseline_models.py
```

* Trains multiple ML models (Logistic Regression, Random Forest, XGBoost, etc.)
* Saves the best pipeline to `models/final_model.pkl`
* Evaluation reports saved in `results/`

### 3. (Optional) Hyperparameter Tuning

```bash
python scripts/05_hyperparameter_tuning.py
```

* Tunes the top models for maximum performance

### 4. Bulk Prediction (CLI)

```bash
python scripts/bulk_predict.py
```

* Runs predictions for `data/test_patients.csv`
* Saves output to `results/test_predictions.csv`

---

## 🌐 Run the App

Launch the Streamlit app:

```bash
streamlit run ui/app.py
```

You’ll see two modes:

1. **Manual Entry**
   Fill in patient details and click **Predict**.
   Example output:

   ```
   Prediction: 1 (Disease)
   Probability: 83%
   ```

2. **Upload CSV**
   Upload a file like `data/test_patients.csv` with multiple records.
   Predictions are shown in a table and downloadable as a CSV.

---

## 📊 Example Prediction

**Sample input (`test_patients.csv`):**

```csv
age,trestbps,chol,thalach,oldpeak,sex,cp,fbs,restecg,exang,slope,ca,thal
54,130,250,160,1.0,1,0,0,1,0,2,0,2
35,120,200,180,0.0,0,2,0,0,0,0,0,1
70,145,300,130,3.5,1,0,1,1,1,2,3,3
```

**Output:**

| age | trestbps | chol | thalach | oldpeak | sex | cp | fbs | restecg | exang | slope | ca | thal | prediction |
| --- | -------- | ---- | ------- | ------- | --- | -- | --- | ------- | ----- | ----- | -- | ---- | ---------- |
| 54  | 130      | 250  | 160     | 1.0     | 1   | 0  | 0   | 1       | 0     | 2     | 0  | 2    | Low Risk   |
| 35  | 120      | 200  | 180     | 0.0     | 0   | 2  | 0   | 0       | 0     | 0     | 0  | 1    | Low Risk   |
| 70  | 145      | 300  | 130     | 3.5     | 1   | 0  | 1   | 1       | 1     | 2     | 3  | 3    | High Risk  |

---

## 🛠️ Tech Stack

* **Python 3.11**
* **pandas / numpy** – data processing
* **scikit-learn** – ML models, pipelines
* **XGBoost** – advanced boosting model
* **Streamlit** – interactive UI
* **joblib** – model serialization

---

## 🚀 Future Improvements

* Add deep learning models (TensorFlow / PyTorch)
* Deploy to cloud (Streamlit Cloud / Heroku / AWS)
* Add explainability with SHAP or LIME
* Create REST API wrapper with FastAPI

---

## 👨‍💻 Author

Developed by [Fady Romany](https://github.com/Eng-fady)
🎯 For learning, experimentation, and educational demonstrations.

```

---