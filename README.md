
# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn based on their demographic information, subscribed services, contract details, and billing information.

The project covers the complete machine learning workflow:

**Data Cleaning → Exploratory Data Analysis → Feature Preprocessing → Logistic Regression → Hyperparameter Tuning → Model Evaluation → Churn Probability Prediction → Web Deployment**

---

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help companies take preventive actions such as targeted offers, improved customer support, and personalized retention strategies.

In this project, a **Logistic Regression classification model** was developed to predict customer churn.

The model uses customer information such as:

- Demographic details
- Customer tenure
- Phone and internet services
- Online security and backup services
- Technical support
- Streaming services
- Contract type
- Payment method
- Monthly charges
- Total charges

The model can also generate a **churn probability**, allowing customers to be ranked according to their likelihood of leaving.

---

## Dataset

The dataset contains:

- **7,043 customer records**
- **21 columns initially**
- **20 predictive features after removing `customerID`**
- Binary target variable: `Churn`

The `customerID` column was removed because it is an identifier and does not provide meaningful predictive information.

### Features

| Feature | Description |
|---|---|
| `gender` | Customer gender |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed |
| `PhoneService` | Whether phone service is subscribed |
| `MultipleLines` | Multiple phone lines subscription |
| `InternetService` | Type of internet service |
| `OnlineSecurity` | Online security subscription |
| `OnlineBackup` | Online backup subscription |
| `DeviceProtection` | Device protection subscription |
| `TechSupport` | Technical support subscription |
| `StreamingTV` | Streaming TV subscription |
| `StreamingMovies` | Streaming movies subscription |
| `Contract` | Contract duration |
| `PaperlessBilling` | Paperless billing status |
| `PaymentMethod` | Customer payment method |
| `MonthlyCharges` | Monthly amount charged |
| `TotalCharges` | Total amount charged |
| `Churn` | Target variable indicating whether the customer churned |

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer behavior and identify patterns associated with churn.

### Categorical Feature Analysis

Churn rates were analyzed across categorical features including:

- Gender
- Partner
- Dependents
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method

Bar plots were used to compare churn rates across different categories.

### Numerical Feature Analysis

Numerical features such as:

- Tenure
- Monthly Charges
- Total Charges

were analyzed against churn using visualizations such as box plots.

The objective of EDA was to understand which customer characteristics are associated with higher or lower churn rates before building the machine learning model.

---

## Data Preprocessing

The dataset contains both numerical and categorical features, requiring different preprocessing techniques.

### Handling `TotalCharges`

`TotalCharges` was converted into a numerical format:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
````

### Categorical Features

Categorical variables were encoded using **OneHotEncoder** with:

```python
OneHotEncoder(
    drop='first',
    handle_unknown='ignore'
)
```

This converts categorical variables into numerical features suitable for Logistic Regression while avoiding redundant dummy variables.

### Numerical Features

Numerical features were standardized using **StandardScaler**.

This is useful because numerical variables such as `tenure`, `MonthlyCharges`, and `TotalCharges` have significantly different scales.

### ColumnTransformer

A `ColumnTransformer` was used to apply the appropriate preprocessing technique to each feature type.

```text
                 Input Data
                     │
          ┌──────────┴──────────┐
          │                     │
   Categorical Features    Numerical Features
          │                     │
   OneHotEncoder          StandardScaler
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
             Logistic Regression
```

The preprocessing and Logistic Regression model were combined into a Scikit-learn `Pipeline`.

---

## Machine Learning Model

### Logistic Regression

Logistic Regression was selected as the primary classification algorithm.

### Why Logistic Regression?

* The target variable is binary.
* It provides probability estimates.
* It is computationally efficient.
* It is relatively interpretable.
* It provides a strong baseline for binary classification.
* Its coefficients can be analyzed to understand feature influence.

The target variable was converted into:

```text
No  → 0
Yes → 1
```

---

## Train-Test Split

The dataset was divided into:

* **75% Training Data**
* **25% Testing Data**

using:

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

This resulted in:

| Dataset      | Records |
| ------------ | ------: |
| Training Set |   5,282 |
| Testing Set  |   1,761 |
| Total        |   7,043 |

The test set contained:

| Class     | Customers |
| --------- | --------: |
| No Churn  |     1,282 |
| Churn     |       479 |
| **Total** | **1,761** |

This demonstrates the class imbalance present in the dataset, making metrics such as precision, recall, F1-score, and ROC-AUC important alongside accuracy.

---

## Hyperparameter Tuning

`GridSearchCV` was used to tune the Logistic Regression model.

The following hyperparameters were evaluated:

```python
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__max_iter': [100, 500, 1000]
}
```

A **5-fold cross-validation** strategy was used.

Multiple evaluation metrics were considered:

```python
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}
```

ROC-AUC was selected as the metric used to determine the best model:

```python
refit='roc_auc'
```

This allows model selection to consider more than just classification accuracy.

---

## Model Performance

The final Logistic Regression model was evaluated on the unseen test set.

| Metric                |     Score |
| --------------------- | --------: |
| **Accuracy**          | **81.4%** |
| **Precision — Churn** |   **68%** |
| **Recall — Churn**    |   **58%** |
| **F1-score — Churn**  |   **63%** |

### Interpretation

**Accuracy — 81.4%**

The model correctly classified approximately 81.4% of customers in the test set.

**Precision — 68%**

Among the customers predicted as churners, approximately 68% actually churned.

**Recall — 58%**

The model successfully identified approximately 58% of the customers who actually churned.

**F1-score — 63%**

The F1-score provides a balance between precision and recall for the churn class and was approximately 0.63.

Because churn is the minority class, accuracy alone does not provide a complete picture of model performance.

---

## Confusion Matrix

The confusion matrix was used to examine the model's classification performance in more detail.

It provides:

* True Negatives
* False Positives
* False Negatives
* True Positives


<img width="1728" height="1466" alt="Image" src="https://github.com/user-attachments/assets/4e03e516-1f83-401f-9899-ed8d6f5a43eb" />


---

## ROC-AUC Curve

The model's predicted probabilities were used to generate the ROC curve.

```python
y_prob = grid.predict_proba(X_test)[:, 1]
```

The ROC curve evaluates the model across different classification thresholds using:

* True Positive Rate
* False Positive Rate


<img width="613" height="470" alt="Image" src="https://github.com/user-attachments/assets/afa398e2-0e84-4eb3-8296-efecf78efbf3" />


> ROC-AUC: 84.1%

---

## Churn Probability Prediction

Instead of returning only a binary prediction, the model can estimate the probability that a customer will churn.

```python
probability = model.predict_proba(input_data)[0][1]
```

For example:

```text
Churn Probability: 73.4%
```

This probability can be useful for prioritizing customers for retention strategies.

A business could potentially focus its resources on customers with higher predicted churn probabilities.

---

## Web Application

A Flask-based web application was developed to allow users to interact with the trained machine learning model.

Users can enter customer information through a web interface, including:

* Demographic information
* Tenure
* Phone and internet services
* Additional subscribed services
* Contract type
* Payment method
* Monthly charges
* Total charges

The application sends the input to the trained machine learning pipeline and returns:

* Churn prediction
* Churn probability

### Application Workflow

```text
User
  │
  ▼
Web Form
  │
  ▼
Flask Backend
  │
  ▼
Trained ML Pipeline
  │
  ├── OneHotEncoder
  │
  ├── StandardScaler
  │
  └── Logistic Regression
  │
  ▼
Prediction Probability
  │
  ▼
Churn Prediction
```

The trained pipeline is saved using Joblib:

```python
joblib.dump(grid.best_estimator_, 'churn_model.pkl')
```

This allows the same preprocessing and trained model to be used by the web application.

---

## Tech Stack

### Programming Language

* Python

### Data Manipulation

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-learn

### Web Development

* Flask
* HTML
* CSS
* JavaScript

### Model Persistence

* Joblib

---

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn_model.pkl
├── customer_churn.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── images/
│   ├── churn_distribution.png
│   ├── categorical_churn.png
│   ├── numerical_churn.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd customer-churn-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

```bash
python app.py
```

### 5. Open the Application

Open the following URL in your browser:

```text
http://127.0.0.1:5000
```

---

## Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
flask
joblib
```

---

## Future Improvements

The current Logistic Regression model provides a strong baseline, but several improvements can be explored.



### 1. Model Comparison

Future versions can compare Logistic Regression with:

* Decision Tree
* Random Forest
* Support Vector Machine
* XGBoost

### 5. Feature Interpretation

Logistic Regression coefficients can be analyzed to understand which customer characteristics have the strongest influence on churn predictions.

---

## Key Learnings

This project provided practical experience with:

* Data cleaning
* Missing-value handling
* Exploratory data analysis
* Categorical feature analysis
* Numerical feature analysis
* One-hot encoding
* Feature scaling
* ColumnTransformer
* Machine learning pipelines
* Logistic Regression
* Cross-validation
* GridSearchCV
* Multiple evaluation metrics
* ROC-AUC analysis
* Classification probability
* Model serialization
* Flask model deployment

---

## Project Highlights

| Category            |                  Result |
| ------------------- | ----------------------: |
| Customer Records    |               **7,043** |
| Initial Features    |                  **21** |
| Predictive Features |                  **20** |
| Training Records    |               **5,282** |
| Testing Records     |               **1,761** |
| Train-Test Split    |               **75:25** |
| Cross-Validation    |              **5-Fold** |
| Model               | **Logistic Regression** |
| Test Accuracy       |               **81.4%** |
| Churn Precision     |                 **68%** |
| Churn Recall        |                 **58%** |
| Churn F1-Score      |                 **63%** |

---

## Author

**Anant Vyas**

Mechanical Engineering Student
Machine Learning & Data Science Enthusiast

---

## Final Summary

This project demonstrates an end-to-end approach to solving a real-world customer churn classification problem.

The workflow combines exploratory data analysis, feature preprocessing, Logistic Regression, cross-validation, hyperparameter tuning, multiple evaluation metrics, probability-based predictions, and a Flask web application.

The final model achieved:

* **81.4% test accuracy**
* **68% precision for the churn class**
* **58% recall for the churn class**
* **63% F1-score for the churn class**



