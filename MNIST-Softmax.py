import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets('MNIST_data/', one_hot=TRUE)

# Placehoders 

x = tf.Placeholder (tf.float32, shape=[None, 784]) #784 is 28 x 28, the pixels of our images


# Variables

## For simplifications purpose, weight and bias is initialized with 0
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# Create Graph

y = tf.matmul(x,W) + b


# Loss Function

## Numbers from 0-9 are 10 possible labels
y_true = tf.placeholder(tf.float32, [None, 10])

## Tensorflow build in cross entropy loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y_true,
    logit=y
))


# Optimizer

optimizer = tf.train.GradientDescent.Optimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)


# Create Session

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for steps in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100) #convinient batch function
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y})

    ## Evaluate Model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true, 1))

    ## We want [True, False, True, ...] to [1,0,1, ...]
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #the average of the 1's and 0's

    ## Example: 
    ## Predicted [3,4] TRUE [3,9] (if someone writes a 9 that looks almost like a 4)
    ## [True, False]
    ## [1.0, 0.0]
    ## Average: 0.5 (percentage how much we are correct, here 50%)
    
    print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))
    ## prints about 0.90 accuracy


