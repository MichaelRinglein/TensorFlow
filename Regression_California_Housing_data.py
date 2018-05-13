# Sorting the Data

import pandas as pd 
housing = pd.read_csv('cal_housing_clean.csv') # The California Housing data


# Train Test Split

x_data = housing.drop('medianHouseValue', axis=1) 
y_val = housing['medianHouseValue'] # this we want to predict

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(
    x_data,
    y_val,
    test_size=0.3 # Training 70%, Testing 30%
)


# Scaling the Feature Data

## Using the MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() #creat instance from MinMaxScaler()
scaler.fit(X_train)

## Creating two dataframes of scaled data
X_train = pd.DataFrame(
    data=scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
) # reset X_train

X_test = pd.DataFrame(
    data=scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# Feature Columns

import tensorflow as tf # finally ;) 
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age, rooms, bedrooms, pop, households, income]

## Creating input function for an estimator
input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    batch_size=10,
    num_epochs=1000,
    shuffle=True
)


# Estimator Model, DNNRegressor
model = tf.estimator.DNNRegressor(
    hidden_units=[6,6,6], # could be increased
    feature_columns=feat_cols
)


# Training the Model

## 25000 steps, can be way more
model.train(input_fn=input_func, steps=10000) #10000 is very little, >25k gets slowly better


# Prediction Function

predict_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test,
    batch_size=10,
    num_epochs=1,
    shuffle=False
)

pred_gen = model.predict(predict_input_func)

predictions = list(pred_gen)
print(predictions)


# Calculating RMSE

final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, final_preds)**0.5)  #results around 50-200k, way to high RSME.

# -> The model can definetely improved








