### Import Libraries


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier 

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


### Preparing data


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


```python
# Preprocessing Feature      ###Switch to label encoder  

#dealing with missing values
features_train['Tenure'] = features_train['Tenure'].fillna(-1)
features_valid['Tenure'] = features_valid['Tenure'].fillna(-1)
features_test['Tenure']  = features_test['Tenure'].fillna(-1)

# Example for Geography
lbe = LabelEncoder()
features_train['Geography'] = lbe.fit_transform(features_train['Geography'])
features_valid['Geography'] = lbe.transform(features_valid['Geography'])
features_test['Geography']  = lbe.transform(features_test['Geography'])

# Example for Gender

ohe = OneHotEncoder()

features_train['Gender'] = ohe.fit_transform(features_train['Gender'])
features_valid['Gender'] = ohe.transform(features_valid['Gender'])
features_test['Gender']  = ohe.transform(features_test['Gender'])

features_train = pd.get_dummies(features_train, columns=categorical_features, drop_first=True)
features_valid = pd.get_dummies(features_valid, columns=categorical_features, drop_first=True)
features_test  = pd.get_dummies(features_test,  columns=categorical_features, drop_first=True)



# Data standardization 
scaler = StandardScaler()

#Define numerical columns
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                   'NumOfProducts', 'EstimatedSalary']

# Fit the scaler ONLY on training data
scaler.fit(features_train[numeric_features])


# Transform all datasets using the same fitted scaler
features_train[numeric_features] = scaler.transform(features_train[numeric_features])
features_valid[numeric_features] = scaler.transform(features_valid[numeric_features])
features_test[numeric_features] = scaler.transform(features_test[numeric_features])

```





## class balance investigation


```python
#investigating to see if the variable distribution is balance
print(data['Exited'].value_counts(normalize=True) * 100) 

#since the result is ~80/20 indicating variable is imbalanced 
model = RandomForestClassifier()

model.fit(features_train, target_train)
```


```python

```
