import tensorflow as tf

t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(tf.reshape(t,[-1,3,3,1]))