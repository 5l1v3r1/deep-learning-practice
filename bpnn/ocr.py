# -*- coding: UTF-8 -*-

import csv
import numpy as np
from numpy import matrix
from math import pow
from collections import namedtuple
import csv
import math
import random
import os
import json

"""
https://www.shiyanlou.com/courses/593/labs/1966/document
https://github.com/Yilia05/DigitRecognize 数据来源
我们构建一个三层的神经网络
其中输入节点400个，因为在我们的手写canvas中格子数是20x20
隐藏层节点是个变量 num_hidden_nodes
输出层有10个节点，分别代表0~9十个数字，哪个数值大就作为我们的预测结果

y0 输入层  1 * 400
theta1 输入-隐藏层权重 
input_layer_bias 输入层偏置向量
y1 隐藏层
theta2 隐藏输出层权重
hidden_layer_bias 隐藏层偏置向量
y2 输出层 10 

 # 初始化神经网络# 一般将所有权值与偏置量置为(-1,1)范围内的随机数，在我们这个例子中，使用(-0.06,0.06)这个范围
 
from collections import namedtuple 非常好用的数据结构，可以看成C的结构体

"""

class OCRNeuralNetwork(object):
    # 学习率
    LEARNING_RATE = 0.1
    # NN文件位置
    NN_file_path = "nn.json"
    # 隐藏层数量
    num_hidden_nodes = 15

    def __init__(self):
        #   使用numpy的vectorize能得到标量函数的向量化版本，这样就能直接处理向量了
        self.sigmoid = np.vectorize(self._sigmoid_scalar)

        # sigmoid 求导函数
        self.sigmoid_prime = np.vectorize(self._sigmoid_scalar_prime)

        # 隐藏层数量
        num_hidden_nodes = self.num_hidden_nodes

        # # 数据集
        # self.data_matrix_list = data_matrix_list
        # self.data_labels_list = data_labels_list


        if os.path.isfile(self.NN_file_path):
            self._load()
        else:
            self.theta1 = self._rand_initialize_weight(400, self.num_hidden_nodes)
            self.theta2 = self._rand_initialize_weight(self.num_hidden_nodes, 10)
            self.input_layer_bias = self._rand_initialize_weight(1, self.num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weight(1, 10)

    def _rand_initialize_weight(self, column, row):
       return [((x*0.12)-0.06) for x in np.random.rand(row,column)]

    def _sigmoid_scalar(self, z):
        """
        sigmoid 激发函数
        :return:
        """
        return 1/(1+math.e ** (-z))

    def _sigmoid_scalar_prime(self, z):
        # 这个导数传入的是输入
        return self.sigmoid(z) * (1-self.sigmoid(z))

    def genTrainData(self, TrainDataList):
        for training_data in TrainDataList:
            y0 = training_data['y0'] # 1 * 400 List?? 这里的数据是整数，与浮点数相距甚远！！
            label = training_data['label'] # 标量, 0,1,2,3,4,5,6,7,8,9 一个s
            with open("data.csv","a") as f:
                f_csv = csv.writer(f)
                f_csv.writerow(y0)
            with open("dataLabels.csv", "a") as f:
                f_csv = csv.writer(f)
                f_csv.writerow([label]) # 这里的保存会影响读取的格式


    def train(self, new=False):
        # 会重新训练神经网络
        if new:
            self.theta1 = self._rand_initialize_weight(400, self.num_hidden_nodes)
            self.theta2 = self._rand_initialize_weight(self.num_hidden_nodes, 10)
            self.input_layer_bias = self._rand_initialize_weight(1, self.num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weight(1, 10)
        # 加载训练集
        data_matrix_list = []
        data_labels_list = []
        with open("data.csv", "r") as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                row = [ float(x) for x in row] # 这里用的是 float
                if row != []:
                    data_matrix_list.append(row)
        with open("dataLabels.csv", "r") as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                # row 是一个 ['3']的数据格式
                row = [int(x) for x in row]
                if row != []:
                    data_labels_list += row
        # 数据集一共个数据，train_indice存储用来训练的数据的序号
        # In Python 3, range returns a lazy sequence object - it does not return a list.
        train_indices_list = list(range(len(data_labels_list)))
        # 打乱训练顺序
        random.shuffle(train_indices_list)
        print("[+]Training data has been loaded successfully!")

        TrainData = namedtuple("TrainData", ['y0', 'label'])
        TrainDataList = [TrainData(data_matrix_list[i], data_labels_list[i]) for i in train_indices_list]
        for training_data in TrainDataList:
            y0 = training_data.y0 # 1 * 400 List??
            label = training_data.label # 标量, 0,1,2,3,4,5,6,7,8,9 一个
            # 前向传播得到结果向量
            y1 = np.dot(np.mat(self.theta1), np.mat(y0).T)
            sum1 = y1 + np.mat(self.input_layer_bias) # 隐藏层的输入向量
            y1 = self.sigmoid(sum1) # 隐藏层的输出向量 num_hidden_nodes * 1

            y2 = np.dot(np.mat(self.theta2), np.mat(y1))
            sum2 = y2 + np.mat(self.hidden_layer_bias) # 输出层的输入向量
            y2 = self.sigmoid(sum2) # 10 * 1

            # 后向传播得到误差向量
            actual_vals = [0] * 10
            actual_vals[label] = 1
            # output_errors = np.mat(actual_vals).T - np.mat(y2) # 10 * 1 原本的，改为如下
            output_errors = np.multiply((np.mat(actual_vals).T - np.mat(y2)), self.sigmoid_prime(sum2)) # 10 * 1
            hidden_errors = np.multiply(np.dot(np.mat(self.theta2).T, output_errors), self.sigmoid_prime(sum1))

            # 更新权重矩阵与偏置向量
            self.theta1 += self.LEARNING_RATE * np.dot(np.mat(hidden_errors), np.mat(y0))
            self.theta2 += self.LEARNING_RATE * np.dot(np.mat(output_errors), np.mat(y1).T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors
        print("[+]Training has been completed successfully!")



    def save(self):
        # 这里 可能是 input_layer_bias 二维列表格式导致的
        json_neural_network = {
            "theta1": [np_mat.tolist()[0] for np_mat in self.theta1], # for i in matrix 按行取，是一个二位数组
            "theta2": [np_mat.tolist()[0] for np_mat in self.theta2],
            "b1": self.input_layer_bias[0].tolist()[0], # tolist()得到二维列表
            "b2": self.hidden_layer_bias[0].tolist()[0]
        }
        with open(self.NN_file_path, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)
        print("[+]Trained NN has been saved!")

    def predict(self, test):
        y1 = np.dot(np.mat(self.theta1), np.mat(test).T)
        y1 = y1 + np.mat(self.input_layer_bias)  # Add the bias
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)  # Add the bias
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        print(results)
        return results.index(max(results))


    def _load(self):
        with open(self.NN_file_path) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = [np.array(nn['b1'][0])]
        self.hidden_layer_bias = [np.array(nn['b2'][0])]
