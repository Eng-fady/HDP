# Code Style Guidelines

To ensure consistency and maintainability across the codebase, please follow these conventions.

---

## General Rules

* Python version: **3.10+**
* Follow **PEP 8** coding style.
* Use **type hints** where possible.
* Keep functions **small and modular** (≤ 30 lines if possible).
* Always document functions with **docstrings**.

---

## Project Structure

```
Heart_Disease_Project/
│── data/                 # Input datasets
│── models/               # Saved ML models
│── notebooks/            # Jupyter notebooks
│── results/              # Generated results, plots, reports
│── scripts/              # Data processing & ML scripts
│── ui/                   # Streamlit application
│── tests/                # Unit tests
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

---

## Python Script Template

Each script must begin with a **standard header block**:

```python
"""
================================================================================
 Script Name   : script_name.py
 Location      : scripts/
 Author        : Full Name
 Date          : YYYY-MM-DD
 Requirements  : pandas, numpy, scikit-learn, etc.
 Input         : path/to/input.csv
 Output        : path/to/output.csv / model.pkl
 Description   :
    - Short summary of purpose.
    - Key steps the script performs.
================================================================================
"""
```

---

## Function Docstrings

* Use the **Google-style docstring format**:

```python
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.

    Args:
        df (pd.DataFrame): Input raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe with imputed values.
    """
```

---

## Code Formatting

* Run **Black** before committing:

  ```bash
  black .
  ```
* Keep imports grouped:

  1. Standard library
  2. Third-party libraries
  3. Local imports

---

## Testing Conventions

* Test file names: `test_<module>.py`
* Use descriptive function names:

  ```python
  def test_preprocess_data_handles_missing_values():
      ...
  ```

---

## Version Control

* Branch naming:

  * `feature/<short-description>`
  * `fix/<short-description>`
  * `doc/<short-description>`
* Commit messages: **present tense & concise**

  * ✅ `Add PCA variance plot`
  * ✅ `Fix missing target column handling`
  * ❌ `Fixed bug in preprocessing`

---

By following this style guide, we ensure the project remains consistent, readable, and professional.
