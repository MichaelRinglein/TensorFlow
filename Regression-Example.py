#Regression Example

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 

#create very large data set, 1 mio points between 0 and 10
x_data = np.linspace(0.0, 10.0, 1000000)

#add noise to the data
noise = np.random.randn(len(x_data))

#graph is modelled by y = mx + b and b = 5
#the true result with m = 0.5 and with some noise in data
y_true = (0.5 * x_data) + 5 + noise

#the goal of the model is now to figure out m
 
#create the dataframe
x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

#concatinate x and y together
my_data = pd.concat([x_df, y_df], axis=1)

#plot a small sample of 250 values of the data (otherwise Kernel might crash)
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.show()

#now let's tensorflow figure out the line

#create a batch of data for training, 8 points a time
batch_size = 8

#first creating variables as random numbers for initiating m and b
m = tf.Variable(0.81)
b = tf.Variable(0.17)

#then creating placeholders for x and y
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

#define graph / model
y_model = m*xph + b

#create loss-function
error = tf.reduce_sum(tf.square(yph-y_model))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learing_rate=0.001)
train= optimizer.minimize(error)

#initializing variables
init = tf.global_variables_initializer()

#run session
with tf.Session() as sess:

    sess.run(init)

    #feeding 1000 batches with 8 values each batch
    batches = 1000 #changing this gives more or less correct result

    for i in range(batches):
        
        rand_ind = np.random.randint(len(x_data), size=batch_size)

        feed = {xph:x_data[rand_ind], yph:y_true[rand_ind]}

        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m,b])

model_m #print model m, this should be close to 0.5

#plot

x_hat = x_data*model_m + model_b

my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(x_data, y_hat, 'r')












