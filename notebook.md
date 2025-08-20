```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
```


```python
data = pd.read_csv('/datasets/Churn.csv')

data.drop(['RowNumber', 'CustomerId' ,'Surname'], axis = 1, inplace = True )
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 11 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   CreditScore      10000 non-null  int64  
     1   Geography        10000 non-null  object 
     2   Gender           10000 non-null  object 
     3   Age              10000 non-null  int64  
     4   Tenure           9091 non-null   float64
     5   Balance          10000 non-null  float64
     6   NumOfProducts    10000 non-null  int64  
     7   HasCrCard        10000 non-null  int64  
     8   IsActiveMember   10000 non-null  int64  
     9   EstimatedSalary  10000 non-null  float64
     10  Exited           10000 non-null  int64  
    dtypes: float64(3), int64(6), object(2)
    memory usage: 859.5+ KB



```python
# Preprocessing Feature
encoder = OneHotEncoder()               
encoder.fit(data[['Geography', 'Gender']])                        
data.head(5)

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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[16], line 14
         10 numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
         11                    'NumOfProducts', 'EstimatedSalary']
         13 # Fit the scaler ONLY on training data
    ---> 14 scaler.fit(features_train[numeric_features])
         16 # Transform all datasets using the same fitted scaler
         17 features_train[numeric_features] = scaler.transform(features_train[numeric_features])


    NameError: name 'features_train' is not defined



```python

```
