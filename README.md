# AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-

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

![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/53003ffcf132d500fcbe3f9525b0df7a3aa25510/Initial%20Given%20Data.png)

The dataset is reasonably balanced between churn and non-churn customers, making accuracy a reliable evaluation metric for measuring model performance.

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
```python
import seaborn as sns
sns.boxplot(x="Churn", y = 'Day Mins',data = churn)
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/25b47a0c759a89661acb81168f1627b6747f77d8/boxplot%20(churn%20vs%20day%20mins).png)

What can we see from the boxplot ?

.The churned custmors have a higher median for Day Mins than not churned customers which means:

. Customer who spent more time calling in a day tend to churn this shows, Days Mins is an important feature

Now, let's see if there is any other feature which has collinearity with Days Mins

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(
    data=churn,
    y_vars=["Day Mins"],
    x_vars=[
        'Account Length', 'VMail Message', 'Eve Mins', 'Night Mins',
        'Intl Mins', 'CustServ Calls', 'Day Calls', 'Day Charge',
        'Eve Calls', 'Eve Charge', 'Night Calls', 'Night Charge',
        'Intl Calls', 'Intl Charge'
    ],
    height=1.5,
    aspect=1
)

plt.show()
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/228735c9f1ab4e7541f04f6908a0516004ea1211/pairplot(%20all%20features).png)

Can we say which feature is highly correlated with Days Mins ?

. Day Charge is highly correlated

. Hence we will just drop either one of them

Now lets see another feature Account Length

```python
sns.boxplot(x = 'Churn', y= 'Account Length', data = churn)
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/0c7d64773d7b4a45a68080245f9d113f7820a856/boxplot(churn%20vs%20AccountLenght).png)
Does the Account Length feature relevant ?

. If we see median for both, its quite similar

. Hence this feature does not have any signficance and can be dropped

Let's plot few more boxplots and check for outliers too
```python
sns.boxplot(x = 'Churn', y = 'Eve Mins',data=churn)
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/fb8ebf8c0bd44fa6465319608593808bc2e6712d/boxplot(churn%20vs%20Eve%20Mins).png)

```python
sns.boxplot( x= 'Churn', y='Intl Plan', data = churn)
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/5cd89ea0942246915858ed47c0a909e89856d068/boxplot(churn%20vs%20Intl%20Plan).png)

Notice how this feature is completely inclined towards one, hence we will not be taking this feature

```python
sns.boxplot(x='Churn',y='CustServ Calls',data = churn)
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/d689113bd2a470af4293711a4e85add720da66f1/boxplot(churn%20vs%20CustServ).png)

```python
sns.boxplot(x='Churn', y='VMail Message', data=churn)
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/0c4be57e399168167b7e5ae2b7fb5f5dff988b6b/boxplot%20(churn%20vs%20VMail%20Message).png)

Again, notice here how there are a lot of outliers for label 1. Hence we will not be taking this feature
```python
sns.boxplot(x='Churn', y='Night Mins', data=churn)
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/364791dc1118e3d0348deb3e3bb19786fcfe969f/boxplot%20(churn%20vs%20Night%20Mins).png)



---

#  Machine Learning Workflow

## 1. Data Preprocessing

- Removed unnecessary/redundant features
  Day Charge and Day Mins are correlated completely

. Similar for Night Charge and Night Mins, and Eve Charge and Eve Mins

. We will be taking one of these features each

. We will also be taking few more features such as "Account Length" and "CustServ Calls"

```python
cols = ['Day Mins', 'Eve Mins', 'Night Mins', 'CustServ Calls', 'Account Length']
y = churn["Churn"]
X = churn[cols]
X.shape
```


- Checked feature correlations
```python
sns.pairplot(data=churn)
```
![image alt](https://github.com/Y7976/AT-T-Telecom-Customer-Churn-Analysis-and-Prediction-/blob/f7754f80f8157144bfb6e03364ab4128f6beaee0/pairplot(whole%20data).png)

Notice how,

. Day Charge and Day Mins are correlated completely

. Similar for Night Charge and Night Mins, and Eve Charge and Eve Mins

. We will be taking one of these features each

. We will also be taking few more features such as "Account Length" and "CustServ Calls"

- Feature scaling using `StandardScaler`
 ```python
  from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
x_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

 Train-validation-test split

 ```python
from sklearn.model_selection import train_test_split

X_tr_cv,X_test,y_tr_cv,y_test = train_test_split(X,y,test_size = 0.2,random_state =1)
X_train , X_val, y_train ,y_val = train_test_split(X_tr_cv ,y_tr_cv, test_size =0.25 ,random_state = 1)
X_train.shape
```

## 2. Model Used

### Logistic Regression

Logistic Regression was selected because:

- It performs well on binary classification problems
- It is interpretable and efficient
- It provides probabilistic outputs for churn prediction
 ```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
```

---

#  Model Performance

The notebook evaluates the model using classification metrics.

```python
model.coef_
```
```python
model.intercept_
```
```python
model.predict(X_train)
### Validation Accuracy
```

- **Accuracy Matric** 
implement our accuracy metric
```python
def accuracy(y_true ,y_pred):
  return np.sum(y_true == y_pred)/y_true.shape[0]
```
```python
accuracy(y_train, model.predict(X_train))
```
```python
accuracy(y_val, model.predict(X_val))
```


---


#. How to Run the project 
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
