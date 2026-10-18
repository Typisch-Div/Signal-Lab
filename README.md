# Customer Churn Predictor

A Flask web app that estimates telecom customer churn risk from account, service, contract, and billing details. It pairs a scikit-learn Logistic Regression pipeline with an interactive **Signal Lab** interface that displays a risk category and estimated probability.

> This is a learning and demonstration project. A prediction is an estimate, not a guarantee or a substitute for business judgment.

## Features

- Interactive form for 19 customer attributes
- Churn risk category and probability estimate
- Preprocessing and model bundled in one saved scikit-learn pipeline
- Responsive interface with grouped customer inputs
- Flask app that can run locally or deploy as a web service

## How it works

1. The notebook loads the Telco Customer Churn CSV and inspects and cleans the data, including converting `TotalCharges` to a numeric value.
2. It removes `customerID`, which is an identifier rather than a predictive customer attribute.
3. It splits the data into training and test sets, using a 75/25 split and `random_state=42`.
4. A scikit-learn `ColumnTransformer` one-hot encodes categorical fields and standardizes numeric fields.
5. A Logistic Regression classifier is trained in a pipeline. `GridSearchCV` with five-fold cross-validation compares configurations and selects a model using ROC-AUC.
6. The selected pipeline is saved as `churn_model.pkl`. The Flask app loads it and returns a churn class and probability for submitted form data.

## Dataset

The included CSV contains 7,043 telecom customer records. `Churn` is the target (`Yes` or `No`). The form collects the 19 input attributes used by the model, such as tenure, subscribed services, contract type, payment method, and charges.

## Run locally

Use Python 3 and run these commands from the project folder:

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Deploy on Render

Create a **Web Service** connected to this repository and use:

| Setting | Value |
| --- | --- |
| Root Directory | Leave blank |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |

The repository root must include `app.py`, `churn_model.pkl`, `requirements.txt`, `templates/`, and `static/`.

## Project files

```text
app.py                         Flask routes and prediction logic
churn_model.pkl                Saved preprocessing and classification pipeline
Data/                          Telco customer churn CSV
Images/                        Notebook analysis and evaluation charts
Notebooks/Customer_Churn.ipynb Data cleaning, analysis, training, and evaluation
templates/index.html           Signal Lab form and results page
static/style.css               Interface styling
requirements.txt               Python dependencies
Procfile                       Production web server command
```

## Tech stack

Python · pandas · NumPy · scikit-learn · Flask · Joblib · HTML · CSS

## Attribution

This repository is adapted from a customer churn project originally created by **Anant Vyas**. The interface and deployment setup have been customized for this version.
