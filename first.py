'''First TensorFlow Code. Yeeey '''

#pylint: disable=E0401
#pylint: disable=C0103
import tensorflow as tf 

a = tf.placeholder(tf.float32, None, 'a')
b = tf.placeholder(tf.float32, None, 'b')

c = a + b

with tf.Session() as sess: 
    ans = sess.run(c, {a:1.0, b:2.0})

    print(ans)