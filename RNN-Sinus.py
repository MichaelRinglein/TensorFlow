import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create Class to create data and create batches

class TimeSeriesData(): 

    def __init__(self, num_points, xmin, xmax): # Creating data, a sinus function

        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)
    
    def ret_true(self, x_series): # Convinience Function
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False): # Generating batches from this data

        # Grab random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)

        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))

        # Create batch time series on the x-axis
        batch_ts = ts_start + np.arange(0.0, steps+1) * self.resolution

        # Create the Y data for the time series x-axis from prev step
        y_batch = np.sin(batch_ts)

        # Formatting for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1) , y_batch[:, 1:].reshape(-1, steps, 1), batch_ts

        else:
            return y_batch[:, :-1].reshape(-1, steps, 1) , y_batch[:, 1:].reshape(-1, steps, 1) # Everything along the rows and everything along the column -1


# Let's create some data

ts_data = TimeSeriesData(250, 0, 10) #250 points between 0 and 10
plt.plot(ts_data.x_data, ts_data.y_true)


# Creating random batches

num_time_steps = 30

y1, y2, ts = ts_data.next_batch(1, num_time_steps, True) # 1 Batch, 30 steps

plt.plot(ts.flatten()[1:], y2.flatten(), '*')

plt.plot(ts_data.x_data, ts_data.y_true, label='Sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='Single Training Instance')
plt.legend()
plt.tight_layout()
plt.show()


# Training data

# Training instance

train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps+1), num_time_steps+1 )

plt.title('A training instance')
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'bo', markersize=15, alpha=0.5, label='Instance')

plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), 'ko', markersize=7, label='Target')
plt.show()

tf.reset_default_graph()


# Constants

# Just one feature, the time series
num_inputs = 1
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1
# learning rate, 0.0001 default, but you can play with this
learning_rate = 0.0001
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 2000
# Size of the batch of data
batch_size = 1


# Placeholders

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])


# RNN Cell Layer

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(
        num_units=num_neurons, 
        activation=tf.nn.relu),
    output_size=num_outputs
)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


# Loss Function

## MSE
loss = tf.reduce_mean(tf.square(outputs - y))


# Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


# Train

train = optimizer.minimize(loss)
init = tf.global_variables_initializer()


# Session

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    for iteration in range(num_train_iterations):
        
        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)

        sess.run(train, feed_dict={X:X_batch, y:y_batch})

        if iteration % 100 == 0:

            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            
            print(iteration, ' \tMSE', mse)

    saver.save(sess, './rnn_time_series_model')

with tf.Session() as sess:

    saver.restore(sess, './rnn_time_series_model')

    X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
    y_pred = sess.run(outputs, feed_dict={X:X_new})

plt.title('Testing the model')

## Training instance
plt.plot(
    train_inst[:-1], 
    np.sin(train_inst[:-1]), 
    'bo', 
    markersize=15, 
    alpha=0.5, 
    label='Training Instance'
)

## Target to predict (correct test values np.sin(train))
plt.plot(
    train_inst[1:],
    np.sin(train_inst[1:]), 
    'ko',
    markersize=10, 
    label='Target' 
)

## Models prediction
plt.plot( 
    train_inst[1:],
    y_pred[0, :, 0],
    'r.',
    markersize=10,
    label='Predictions'
)

plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()

