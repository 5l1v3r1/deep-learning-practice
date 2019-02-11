# encoding:utf-8
import time
import numpy as np

def exeTime(func):
    def wrapper(*args, **kwargs):
         start = time.time()
         r = func(*args, **kwargs)
         end = time.time()
         return r, end-start
    return wrapper

def loadDataSet(filename):
    """
     从文件中获取数据，在《机器学习实战中》，数据格式如下
    "feature1 TAB feature2 TAB feature3 TAB label"

    :param filename 文件名:
    :return X 训练样本集矩阵, Y 标签集矩阵:
    """
    numFeat = len(open(filename).readline().split('\t')) - 1
    X = []
    y = []
    f = open(filename)
    for line in f.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        X.append(lineArr)
        y.append(float(curLine[-1]))
    return np.mat(X),np.mat(y).T

def h(theta, x):
    """预测函数

        Args:
            theta: 相关系数矩阵
            x: 特征向量

        Returns:
            预测结果
    """
    return (theta.T * x)[0, 0]

def J(theta, X, y):
    """代价函数

        Args:
            theta: 相关系数矩阵
            X: 样本集矩阵
            y: 标签集矩阵

        Returns:
            预测误差（代价）
    """
    m = len(X)
    return (X*theta-y).T*(X*theta-y)/(2*m)



@exeTime
def bgd(rate, maxLoop, epsilon, X, y):
    """批量梯度下降法

    Args:
        rate: 学习率
        maxLoop: 最大迭代次数
        epsilon: 收敛精度
        X: 样本矩阵
        y: 标签矩阵

    Returns:
        (theta, errors, thetas), timeConsumed
    """
    m,n = X.shape
    # 初始化theta
    theta = np.zeros((n,1))
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count<=maxLoop:
        if(converged):
            break
        count = count + 1
        for j in range(n):
            deriv = (y-X*theta).T*X[:, j]/m
            theta[j,0] = theta[j,0]+rate*deriv
            thetas[j].append(theta[j,0])
        error = J(theta, X, y)
        errors.append(error[0,0])
        # 如果已经收敛
        if(error < epsilon):
            converged = True
    return theta,errors,thetas
"""
可以看到，bgd 运行的并不慢，这是因为在 regression 程序中，我们采用了向量形式计算 θ，计算机会通过并行计算的手段来优化速度。
"""

@exeTime
def sgd(rate, maxLoop, epsilon, X, y):
    """随机梯度下降法
    Args:
        rate: 学习率
        maxLoop: 最大迭代次数
        epsilon: 收敛精度
        X: 样本矩阵
        y: 标签矩阵
    Returns:
        (theta, error, thetas), timeConsumed
    """
    m,n = X.shape
    # 初始化theta
    theta = np.zeros((n,1))
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count <= maxLoop:
        if(converged):
            break
        count = count + 1
        errors.append(float('inf'))
        for i in range(m):
            if(converged):
                break
            diff = y[i,0]-h(theta, X[i].T)
            for j in range(n):
                theta[j,0] = theta[j,0] + rate*diff*X[i, j]
                thetas[j].append(theta[j,0])
            error = J(theta, X, y)
            errors[-1] = error[0,0]
            # 如果已经收敛
            if(error < epsilon):
                converged = True
    return theta, errors, thetas