import numpy as np
import tensorflow as tf
import datetime

# Processing Units logs
log_device_placement = True

# Num of multiplications to perform
n = 10

# Create random large matrix
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')
C = np.random.rand(10000, 10000).astype('float32')
D = np.random.rand(10000, 10000).astype('float32')

# Create a graph to store results
c1 = []
c2 = []

def matpow(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

'''
Single GPU computing
'''
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c1) 

t1_1 = datetime.datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    sess.run(sum, {a:A, b:B})
t2_1 = datetime.datetime.now()


'''
Multi GPU computing
'''
# GPU:0 computes A^n
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(a, n))

# GPU:1 computes B^n
with tf.device('/gpu:1'):
    b = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c2) #Addition of all elements in c2, i.e. A^n + B^n

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_2 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2_1-t1_1))
print("Multi GPU computation time: " + str(t2_2-t1_2))
