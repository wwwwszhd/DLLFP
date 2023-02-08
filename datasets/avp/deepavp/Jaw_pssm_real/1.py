import tensorflow.compat.v1 as tf
import scipy.io as sio
import load_data

data, sequence_length, label = load_data.load_train_data()

matrix = sio.loadmat('./data/pssm.mat')['pssm']
initializer_filters = tf.reshape(tf.constant(matrix, dtype=tf.float32), [1, 20, 1, 20])
initializer_biases = tf.constant_initializer(0)
filters = tf.get_variable('filters', initializer=initializer_filters, dtype=tf.float32)
biases = tf.get_variable('biases', [20], initializer=initializer_biases, dtype=tf.float32, trainable=False)

input = tf.reshape(data, [951, 107, 20, 1])
input = tf.cast(input,dtype=tf.float32)
temp = tf.nn.conv2d(input, filters, strides=[1, 1, 20, 1], padding='SAME')
temp_b = tf.nn.bias_add(temp, biases)
conv_pssm = temp_b  # shape= [batch_size, 107, 1, 20]
print(conv_pssm)