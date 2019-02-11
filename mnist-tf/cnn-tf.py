# encoding:utf-8
# https://juejin.im/entry/5b079b346fb9a07a9c04a38a
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# MNIST_data 代表当前程序文件所在的目录中，用于存放MNIST数据的文件夹，如果没有则新建，然后下载．
mnist = input_data.read_data_sets("data",one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
# 可视化测试图片
#获取第二张图片
image = mnist.train.images[1,:]
#将图像数据还原成28*28的分辨率
image = image.reshape(28,28)
#打印对应的标签
print(mnist.train.labels[1])
plt.figure()
plt.imshow(image)
plt.show()

# None　代表图片数量未知
input = tf.placeholder(tf.float32,[None,784])
# 将input 重新调整结构，适用于CNN的特征提取
input_image = tf.reshape(input,[-1,28,28,1])
# y是最终预测的结果
y = tf.placeholder(tf.float32,[None,10])

# 卷积层
# input 代表输入，filter 代表卷积核
def conv2d(input,filter):
    return tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')

# 池化层
def max_pool(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 初始化卷积核或者是权重数组的值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    # tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
    return tf.Variable(initial)

# 初始化bias的值
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))

#[filter_height, filter_width, in_channels, out_channels]
#定义了卷积核
filter = [3,3,1,32]

filter_conv1 = weight_variable(filter)
b_conv1 = bias_variable([32])
# 创建卷积层，进行卷积操作，并通过Relu激活，然后池化
h_conv1 = tf.nn.relu(conv2d(input_image,filter_conv1)+b_conv1)
h_pool1 = max_pool(h_conv1)

# 全连接层和输出层
h_flat = tf.reshape(h_pool1,[-1,14*14*32]) # h_flat 是将　pool 后的卷积核全部拉平成一行数据，便于和后面的全连接层进行数据运算．

W_fc1 = weight_variable([14*14*32,784])
b_fc1 = bias_variable([784])
h_fc1 = tf.matmul(h_flat,W_fc1) + b_fc1

W_fc2 = weight_variable([784,10])
b_fc2 = bias_variable([10])

y_hat = tf.matmul(h_fc1,W_fc2) + b_fc2 # y_hat 是整个神经网络的输出层，包含 10 个结点．

# 代价函数采用了 cross_entropy，显然，整个模型输出的值经过了 softmax 处理，将输出的值换算成每个类别的概率．
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat ))

# 训练神经网络
# 在这里，定义了一个梯度下降的训练器，学习率是0.01．
# 我们只需要知道，train_step在每一次训练后都会调整神经网络中参数的值，以便　cross_entropy 这个代价函数的值最低，也就是为了神经网络的表现越来越好
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# 上面代码的目的是定义准确率，我们会在后面的代码中周期性地打印准确率

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):

        batch_x,batch_y = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={input:batch_x,y:batch_y})
            print("step %d,train accuracy %g " %(i,train_accuracy))

        train_step.run(feed_dict={input:batch_x,y:batch_y})

    print("test accuracy %g " % accuracy.eval(feed_dict={input:mnist.test.images,y:mnist.test.labels}))

# 我们的　epoch 是　10000 次，也就是说需要训练10000个周期．每个周期训练都是小批量训练　50 张，然后每隔　１00 个训练周期打印阶段性的准确率．





