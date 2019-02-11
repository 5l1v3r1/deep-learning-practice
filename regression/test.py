# encoding:utf-8
# https://blog.csdn.net/cjf1699/article/details/79223392
import numpy as np

a=[[1,2,3],[4,5,6]]
a=np.array(a)
print(type(a))
print(a)
print(a.transpose())    # 转置换          # 两者都有transpose方法
b=np.array([1,2,2])
print(b)
print(a*b)

c=np.mat([[1,2,3],[4,5,6]])    # mat，将列表矩阵化
print(c)
d=np.mat([1,2,2])

print(np.shape(b))
print(np.shape(d))
print(c*d.transpose())