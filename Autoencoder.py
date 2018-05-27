import numpy as np
import matplotlib.pyplot as plt


# Get some data
from sklearn.datasets import make_blobs
data = make_blobs(
    n_samples=100, 
    n_features=3, 
    centers=2,
    random_state=101 
) # n_features = 3, 3 dimensional data


# Scaling data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[0])

data_x = scaled_data[:, 0] # All rows in column 0
data_y = scaled_data[:, 1] # All rows in column 1
data_z = scaled_data[:, 2] # All rows in column 2


# If we now use Linear Autoencoder to transform the 3D data to 2D, can we still see the seperation of the two blombs? 
# Let's try

import tensorflow as tf 
from tensorflow.contrib.layers import fully_connected

num_inputs = 3 # Since we have 3 Dimensions worth of data
num_hidden = 2
num_outputs = num_inputs # Autoencoder, so must be same as input

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden = fully_connected(X, num_hidden, activation_fn=None) # No activation function since Linear Autoencoder
output = fully_connected(hidden, num_outputs, activation_fn=None)

loss = tf.reduce_mean(tf.square(X-output))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

num_steps = 1000

with tf.Session() as sess: 

    sess.run(init)

    for iteration in range(num_steps): 

        sess.run(train, feed_dict={X:scaled_data})

    output_2d = hidden.eval(feed_dict={X:scaled_data}) # We want to get to the 2D data in the hidden layer

plt.scatter(output_2d[:, 0], output_2d[:, 1], c=data[1])
plt.show()

