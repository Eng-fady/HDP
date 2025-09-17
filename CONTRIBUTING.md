# Contributing to Heart Disease Risk Predictor

Thank you for your interest in contributing to this project!
We welcome contributions of all kinds: bug fixes, new features, documentation improvements, or performance enhancements.

---

## How to Contribute

### 1. Fork & Clone

* Fork this repository to your own GitHub account.
* Clone it locally:

  ```bash
  git clone https://github.com/<your-username>/Heart_Disease_Project.git
  cd Heart_Disease_Project
  ```

### 2. Set Up Development Environment

* Create a virtual environment:

  ```bash
  python -m venv venv
  source venv/bin/activate    # Linux/Mac
  venv\Scripts\activate       # Windows
  ```
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

### 3. Run Pre-Commit Checks

* Format code:

  ```bash
  black .
  ```
* Lint code:

  ```bash
  flake8 scripts/ ui/
  ```

### 4. Make Your Changes

* Add new functionality or fix issues.
* Ensure all scripts are documented using the **standardized header block** and function-level docstrings.

### 5. Test

* Run unit tests:

  ```bash
  pytest
  ```
* Verify that the Streamlit app runs:

  ```bash
  streamlit run ui/app.py
  ```

### 6. Submit a Pull Request

* Push changes to your fork.
* Open a Pull Request (PR) to the `main` branch of this repo.
* In the PR description:

  * Clearly describe the motivation for the change.
  * Link any related issues.
  * Include before/after screenshots for UI changes.

---

## Testing Guidelines

* Tests should be placed in a `tests/` folder.
* Use **pytest** with descriptive test names.
* Ensure that new functionality is covered by tests.

---

## Contribution Recognition

All accepted contributors will be added to the **Contributors** section in the README.
