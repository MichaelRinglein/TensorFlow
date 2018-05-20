import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


# Helper Functions

## Initializing random weights
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

## Initializing random biases
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

## 2-D convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

## Pooling layer
def max_pool_2by2(x):
    return tf.nn.max_pool(
        x, 
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
        )

## return actual convolutional layer with relu
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

## normal fully connected layer
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# Placeholders

x = tf.placeholder(tf.float32, shape=[None, 784]) #28x28=784
y_true = tf.placeholder(tf.float32, shape=[None, 10]) #10 possible results, numbers 0-9


# Layers

x_image = tf.reshape(x, [-1, 28, 28, 1])

## Convolutional layers

convo_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[6, 6, 32, 64])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7*7*64]) #(28/2)/2=7
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 10)


# Loss Function

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y_true,
    logits=y_pred
))


# Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.01) # bad learning rate
train = optimizer.minimize(cross_entropy)


# Initialize Variables

init = tf.global_variables_initializer()


# Session

with tf.Session() as sess:

    sess.run(init) 

    for i in range(500):

        batch_x, batch_y = mnist.train.next_batch(50)

        sess.run(train, feed_dict={
            x:batch_x,
            y_true:batch_y,
            hold_prob:0.5
        })

    ## Message every 100 steps
    if i%100 == 0:

        print('Currently on step {}'.format(i))
        print('Accuracy is:')

        ## Testing train model
        matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

        acc = tf.reduce_mean(tf.cast(matches, tf.float32))

        print(sess.run(acc, feed_dict={
            x:mnist.test.images,
            y_true:mnist.test.labels,
            hold_prob:1.0
        }))
        print('\n')