# encoding:utf-8
# 要求，这个版本的keras低一些
import keras
print(keras.__version__) # 2.2.4
version = keras.__version__.split(".")[0]
version = int(version)
# 构建深度卷积神经网络
# https://zhuanlan.zhihu.com/p/26078299
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random

import string
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)
# width,height用于指定图片宽高
# n_len, n_class 验证码位数和候选字符种类个数

def gen(batch_size=32):
    '''
    一批的数量:32
    :param batch_size:
    :return:
    '''
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8) # 无符号整数64位
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)] # 一个长度为4的列表，每个元素是一个二维张量
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y
# yield generator 用法

from keras.models import *
from keras.layers import *



input_tensor = Input((height, width, 3)) # 得到一个张量

x = input_tensor
for i in range(4):
    # 二维卷积层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。例如input_shape = (3,128,128)代表128*128的彩色RGB图像
    if version == 2:
        x = Convolution2D(32 * 2 ** i, (3,3), activation='relu')(x)  # Keras 函数式 API
        x = Convolution2D(32 * 2 ** i, (3,3), activation='relu')(x)  # 卷积核数目，卷积核行数，卷积核列数
        x = MaxPooling2D((2, 2))(x)  # pool_size：长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半
        # 为空域信号施加最大值池化
    else:
        x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)  # Keras 函数式 API
        x = Convolution2D(32 * 2 ** i, 3, 3, activation='relu')(x)  # 卷积核数目，卷积核行数，卷积核列数
        x = MaxPooling2D((2, 2))(x)  # pool_size：长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半
        # 为空域信号施加最大值池化



# 我们可以看到最后一层卷积层输出的形状是 (1, 6, 256)，已经不能再加卷积层了。
x = Flatten()(x) # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
x = Dropout(0.25)(x) # dropout(x, level, seed=None) 随机将x中一定比例的值设置为0，并放缩整个tensor
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)] # Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。
# 经过上面的步骤，我们将它 Flatten，然后添加 Dropout ，尽量避免过拟合问题，最后连接四个分类器，每个分类器是36个神经元，输出36个字符的概率。
if version ==2 :
    model = Model(inputs=input_tensor, outputs=x)
else:
    model = Model(input=input_tensor, output=x)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
# loss：目标函数，为预定义损失函数名或一个目标函数，参考目标函数
# optimizer：优化器，为预定义优化器名或优化器对象，参考优化器
# metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']


# 可视化模型
# 这里需要使用 pydot 这个库，以及 graphviz 这个库
# from keras.utils.visualize_util import plot
# from IPython.display import Image
#
# plot(model, to_file="model.png", show_shapes=True)
# Image('model.png')


# 模型训练
# windows 平台好像不支持多线程~GG
model.fit_generator(gen(), steps_per_epoch=1600, epochs=5,
                    workers=1, use_multiprocessing=False,
                    validation_data=gen(), validation_steps=40)
# Fits the model on data generated batch-by-batch 逐批 by a Python generator.
# samples_per_epoch 类似指定样本数的功能  steps_per_epoch = samples_per_epoch/batch_size Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed.
# nb_epoch total number of iterations on the data，越大越准确，但是耗时也越大
# nb_worker: maximum number of processes to spin up 计算机术语，启动
# pickle_safe: if True, use process based threading. Note that because this implementation relies on multiprocessing, you should not pass non picklable arguments to the generator as they can't be passed easily to children processes.
# validation_data this can be either
## a generator for the validation data
## a tuple (inputs, targets)
## a tuple (inputs, targets, sample_weights).
# nb_val_samples: only relevant if validation_data is a generator. number of samples to use from validation generator at the end of every epoch 新世代，纪元.

import pickle
import os
if os.path.exists("model.pickle"):
    with open("model.pickle","rb") as f:
        model = pickle.load(f)
else:
    with open("mode.pickle", "wb") as f:
        pickle.dump(model, f)


# 训练完成后验证

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]# 取出array最大的值对应的索引，默认索引从0开始
    # axis:返回沿轴axis最大值的索引。默认情况下，索引的是平铺的数组，否则沿指定的轴。
    return ''.join([characters[x] for x in y])


X, y = next(gen(1))
y_pred = model.predict(X)
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')

# 评估准确度，这里用到了一个库叫做 tqdm，它是一个进度条的库
# from tqdm import tqdm
# def evaluate(model, batch_num=20):
#     batch_acc = 0
#     generator = gen()
#     for i in tqdm(range(batch_num)):
#         X, y = next(generator)
#         y_pred = model.predict(X)
#         y_pred = np.argmax(y_pred, axis=2).T
#         y_true = np.argmax(y, axis=2).T
#         batch_acc += np.mean(map(np.array_equal, y_true, y_pred))
#     return batch_acc / batch_num
#
# evaluate(model)