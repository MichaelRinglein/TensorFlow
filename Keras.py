# Getting the data

from sklearn.datasets import load_wine
wine_data = load_wine()
#wine_data.keys() # checking the keys
#print(wine_data['DESCR']) # get a feeling for the data


# feature data & labels

feat_data = wine_data['data']
labels = wine_data['target']


# Train-Test-Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    feat_data,
    labels,
    test_size=0.3,
    random_state=101
)


# Normalizing data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)


# Build the model with Keras

import tensorflow as tf
from tensorflow.contrib.keras import models 


# Create a model with Keras

dnn_keras_model = models.Sequential() 


# Creating the layers

from tensorflow.contrib.keras import layers

## Adding layers
dnn_keras_model.add(layers.Dense(units=13, input_dim=13, activation='relu')) # First layer, so we need to define input dimensions

dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=13, activation='relu'))

dnn_keras_model.add(layers.Dense(units=3, activation='softmax')) 


# Loss function, optimizer etc

from tensorflow.contrib.keras import losses, optimizers, metrics, activations

dnn_keras_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

dnn_keras_model.fit(scaled_x_train, y_train, epochs=50)