### Import Libraries


```python
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score, roc_auc_score
```


```python
data = pd.read_csv('/datasets/Churn.csv')

data.drop(['RowNumber', 'CustomerId' ,'Surname'], axis = 1, inplace = True )
print(data.shape)
display(data.head(10))
```

    (10000, 11)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1.0</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8.0</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2.0</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>645</td>
      <td>Spain</td>
      <td>Male</td>
      <td>44</td>
      <td>8.0</td>
      <td>113755.78</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>149756.71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>822</td>
      <td>France</td>
      <td>Male</td>
      <td>50</td>
      <td>7.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>10062.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>376</td>
      <td>Germany</td>
      <td>Female</td>
      <td>29</td>
      <td>4.0</td>
      <td>115046.74</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>119346.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>501</td>
      <td>France</td>
      <td>Male</td>
      <td>44</td>
      <td>4.0</td>
      <td>142051.07</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>74940.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>684</td>
      <td>France</td>
      <td>Male</td>
      <td>27</td>
      <td>2.0</td>
      <td>134603.88</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>71725.73</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Preprocessing data


```python
#fill missing data with -1
#check if there are any NaN data left
data['Tenure'].fillna(-1, inplace=True)
print(data['Tenure'].isna().sum())

#assign ohe for Gender. 0 = Female, 1 = Male
#set drop_first to avoid dummy trap
data['Gender'] = pd.get_dummies(data['Gender'], drop_first=True)

#assign ohe for geography
data = pd.get_dummies(data, columns=['Geography'], prefix='Geography')

#displaying data
display(data.head(5))
```

    0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
      <th>Geography_France</th>
      <th>Geography_Germany</th>
      <th>Geography_Spain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>0</td>
      <td>42</td>
      <td>2.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>0</td>
      <td>41</td>
      <td>1.0</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>0</td>
      <td>42</td>
      <td>8.0</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699</td>
      <td>0</td>
      <td>39</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850</td>
      <td>0</td>
      <td>43</td>
      <td>2.0</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### Spliting data


```python
#split the data
data_temp, data_test = train_test_split(data, test_size=0.20, random_state=12345)
data_train, data_valid = train_test_split(data, test_size=0.25, random_state=12345)

features_train = data_train.drop(['Exited'], axis = 1)
target_train = data_train['Exited']

features_valid = data_valid.drop(['Exited'],axis = 1)
target_valid = data_valid['Exited']

features_test = data_test.drop(['Exited'],axis =1)
target_test = data_test['Exited']

```

## class balance investigation


```python
#investigating to see if the variable distribution is balance
print(data['Exited'].value_counts(normalize=True) * 100) 
```

    0    79.63
    1    20.37
    Name: Exited, dtype: float64


The finding indicating that our target class have an imbalance distribution, which can be probablematic to our training model and prediction  

### Train baseline model
this training purpose is to show how imbalance target distribution could affect the training model


```python
#train baseline model 
def evaluate_model(model, features_train, target_train, features_valid, target_valid):
    model.fit(features_train, target_train)
    
    pred_valid = model.predict(features_valid)
    proba_valid = model.predict_proba(features_valid)[:, 1]
    
    f1 = f1_score(target_valid, pred_valid)
    auc = roc_auc_score(target_valid, proba_valid)
    
    return f1, auc

# Logistic Regression
log_reg = LogisticRegression(random_state=12345, max_iter=1000)
f1_lr, auc_lr = evaluate_model(log_reg, features_train, target_train, features_valid, target_valid)
print("Logistic Regression F1 score:", f1_lr,", AUC-ROC score:", auc_lr)

# Random Forest
rf = RandomForestClassifier(random_state=12345, n_estimators=100)
f1_rf, auc_rf = evaluate_model(rf, features_train, target_train, features_valid, target_valid)
print("Random Forest F1 score:", f1_rf, ", AUC-ROC score:", auc_rf)
```

    Logistic Regression F1 score: 0.10784313725490195 , AUC-ROC score: 0.6717757009345795
    Random Forest F1 score: 0.559423769507803 , AUC-ROC score: 0.8504007039071604


#### Finding from base model training 
##### Logistic Regression: 

F1 score is way too low.

AUC-ROC score is decent separation, but the model struggles to catch the minority class.

##### Random Forest:

F1 is a bit much better, close to the project requirement of ≥ 0.59.

AUC-ROC score is strong performance at ranking churn vs non-churn.

##### This confirms what we suspected: class imbalance is hurting the models, especially Logistic Regression. Random Forest already does better, but we can push it higher with imbalance handling.

### Handling Imbalance Class


```python

# Logistic Regression with balanced class weights
lr_balance = LogisticRegression(random_state=12345, class_weight="balanced")
f1_lr_bal, auc_lr_bal = evaluate_model(lr_balance, features_train, target_train, features_valid, target_valid)

print("Logistic Regression (balanced) -> F1:", f1_lr_bal, " | AUC-ROC:", auc_lr_bal)

# Random Forest with balanced class weights
rf_balance = RandomForestClassifier(random_state=12345, class_weight="balanced")
f1_rf_bal, auc_rf_bal = evaluate_model(rf_balance, features_train, target_train, features_valid, target_valid)

print("Random Forest (balanced) -> F1:", f1_rf_bal, " | AUC-ROC:", auc_rf_bal)

```

    Logistic Regression (balanced) -> F1: 0.45555555555555555  | AUC-ROC: 0.7099074932819672
    Random Forest (balanced) -> F1: 0.5480769230769231  | AUC-ROC: 0.8498033340467526


The result show a better distribution but yet still not meet the goal that we're looking for
In this case we have to upsampling the minor classes 

#### Upsampling data


```python
# Combine features + target for upsampling
train_upsampled = pd.concat([features_train, target_train], axis=1)

# Separate classes
exited = train_upsampled[train_upsampled['Exited'] == 1]
not_exited = train_upsampled[train_upsampled['Exited'] == 0]

# Upsample exited customers
exited_upsampled = resample(exited, replace=True, n_samples=len(not_exited), random_state=12345)

# New balanced training set
upsampled = pd.concat([not_exited, exited_upsampled])

features_train_upsampled = upsampled.drop('Exited', axis=1)
target_train_upsampled = upsampled['Exited']

```

#### Training upsampled data with two models


```python
# Train Logistic Regression on upsampled data
lr_upsampled = LogisticRegression(random_state=12345)
lr_upsampled.fit(features_train_upsampled, target_train_upsampled)

pred_valid_upsampled_lr = lr_upsampled.predict(features_valid)
proba_valid_upsampled_lr = lr_upsampled.predict_proba(features_valid)[:, 1]

print("Logistic Regression F1 score (upsampled):", f1_score(target_valid, pred_valid_upsampled_lr), 
      " , AUC-ROC score:", roc_auc_score(target_valid, proba_valid_upsampled_lr))


# Train Random Forest on upsampled data
rf_upsampled = RandomForestClassifier(random_state=12345)
rf_upsampled.fit(features_train_upsampled, target_train_upsampled)

pred_valid_upsampled = rf_upsampled.predict(features_valid)
proba_valid_upsampled = rf_upsampled.predict_proba(features_valid)[:, 1]

print("Random Forest F1 score (upsampled):", f1_score(target_valid, pred_valid_upsampled), 
      " , AUC-ROC score:", roc_auc_score(target_valid, proba_valid_upsampled))

```

    Logistic Regression F1 score (upsampled): 0.4559457794208256  , AUC-ROC score: 0.7108863047252146
    Random Forest F1 score (upsampled): 0.6042105263157895  , AUC-ROC score: 0.8445497134432


### Conclusion



After testing two models to find which is better in predicting customer exiting, here's my conclusion: 
##### Logistic Regression 
Even with upsampling, only reached an F1 score of 0.46, showing it struggled with the churn problem. 
##### Random Forest 
the model performed much better, achieving an F1 score of 0.60 and an AUC-ROC of 0.85, which meets the project goal.

This means Random Forest is better at finding patterns in the data and can identify customers likely to leave with stronger accuracy. Overall, handling class imbalance and using a tree-based model were key to improving predictions and creating a useful tool for customer retention.
