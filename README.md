# AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-
#  Customer Churn Prediction using Logistic Regression

Predicting customer churn is one of the most important business problems in the telecom industry. This project builds a machine learning model using **Logistic Regression** to identify customers who are likely to leave a telecom service provider.

The project includes:

- Exploratory Data Analysis (EDA)
- Feature selection and preprocessing
- Logistic Regression model training
- Model evaluation and validation
- Insights from telecom customer behavior data

---

#  Project Overview

Customer churn refers to customers discontinuing a company's service. Retaining existing customers is often more cost-effective than acquiring new ones.

In this project, a churn prediction system was developed using a telecom dataset containing customer usage patterns, service plans, call statistics, and customer support interactions.

The model helps identify patterns that indicate whether a customer is likely to churn.

---

#  Dataset Information

The dataset contains telecom customer information with features such as:

| Feature | Description |
|---|---|
| Account Length | Number of days the account has been active |
| VMail Message | Number of voicemail messages |
| Day Mins | Total daytime minutes used |
| Eve Mins | Total evening minutes used |
| Night Mins | Total night minutes used |
| Intl Mins | Total international minutes used |
| CustServ Calls | Number of customer service calls |
| Intl Plan | Whether customer has international plan |
| VMail Plan | Whether customer has voicemail plan |
| Day Calls / Eve Calls / Night Calls | Number of calls during different periods |
| Charges | Corresponding charges for usage |
| State | Customer state |
| Area Code | Area code |
| Churn | Target variable (0 = No Churn, 1 = Churn) |

### Dataset Shape

- **Rows:** 5700
- **Columns:** 21

---

#  Technologies Used

- Python
- Google Colab Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

#  Exploratory Data Analysis (EDA)

Several visualizations and statistical analyses were performed to understand customer behavior and feature relationships.

### Key Findings

- Customers with higher **Day Minutes** usage showed higher churn probability.
- Features like **Day Charge** were highly correlated with **Day Minutes**.
- Certain features contained significant outliers.
- Some features showed very low predictive importance and were excluded.
- The dataset appeared reasonably balanced between churned and non-churned customers.

### Visualizations Included

- Count plots
- Box plots
- Correlation heatmaps
- Feature distribution analysis

---

#  Machine Learning Workflow

## 1. Data Preprocessing

- Removed unnecessary/redundant features
- Checked feature correlations
- Feature scaling using `StandardScaler`
- Train-validation-test split

## 2. Model Used

### Logistic Regression

Logistic Regression was selected because:

- It performs well on binary classification problems
- It is interpretable and efficient
- It provides probabilistic outputs for churn prediction

---

#  Model Performance

The notebook evaluates the model using classification metrics.

### Validation Accuracy

- **Accuracy:** ~71%

Additional evaluation methods may include:

- Confusion Matrix
- Precision
- Recall
- F1 Score

---

```

---

#  How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

## 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
```

## 3. Run the Notebook

```bash
jupyter notebook
```

Open:

```bash
AT&T_Analysis_Project.ipynb
```

---

#  Business Insights

This project demonstrates how machine learning can help telecom companies:

- Detect customers at risk of leaving
- Improve customer retention strategies
- Reduce revenue loss
- Enhance customer satisfaction
- Support data-driven decision making

---

#  Future Improvements

Possible future enhancements:

- Try advanced models like Random Forest, XGBoost, or Neural Networks
- Hyperparameter tuning
- Deploy as a web application using Flask or Streamlit
- Add real-time prediction API
- Improve feature engineering

---

#  Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork the repository and submit a pull request.

---

#  License

This project is open-source and available under the MIT License.

---
