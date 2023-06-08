import GPy
import numpy as np
import pandas as pd


train_filename = 'data/FE_all.csv'

all_data = pd.read_csv(train_filename)
X = np.array(all_data.iloc[:, 0:4])
# indices = ['Q/-3_9','Q/0.5_9']
indices = ['Q/-3_9','f9']
Y = np.array(all_data.loc[:, indices])
# Y = np.array(all_data.iloc[:, 8:10])

from sklearn.model_selection import train_test_split

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(train_data, train_targets, test_size=13, random_state=0)

mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std # 归一化处理？
#
# # 测试数据必须以训练数据的均值和标准差进行变换，这样两者的变换才一致
X_test -= mean
X_test /= std


# LCM 多核
p = X_train.shape[1]
n = X_train.shape[0]
po = y_train.shape[1]
no = y_train.shape[0]


def lcm(input_dim=4, num_outputs=4):
    p = input_dim
    K1 = GPy.kern.Bias(p)
    K2 = GPy.kern.RBF(p)
    K3 = GPy.kern.Matern32(p)
    lcm = GPy.util.multioutput.LCM(input_dim,
                                   num_outputs,
                                   kernels_list=[K1, K2, K3],
                                   W_rank=2)
    return lcm


def XY(X_train, y_train, num_output):
    # 输出时预测点X格式重构
    X = [X_train]
    Y = [y_train[:, 0].reshape(-1, 1)]
    k = num_output
    i = 2
    while i <= k:
        X.append(X_train)
        Y.append(y_train[:, i - 1].reshape(-1, 1))
        i += 1
    return X, Y


X, Y = XY(X_train, y_train, po)
# ICM 单核
# K= GPy.util.multioutput.ICM(input_dim=p, num_outputs=po, kernel=Matern32(4), W_rank=2)
# K = icm()
K = lcm(p, po)
# 模型构建与参数优化
m = GPy.models.GPCoregionalizedRegression(X, Y, kernel=K)
# m = GPy.models.GPCoregionalizedRegression([X_train, X_train], [Y1, Y2], kernel=K)
# m['.*Mat32.var'].constrain_fixed(1000.)  # For this kernel, B.kappa encodes the variance now.
m['.*ICM.*var'].unconstrain()
m['.*ICM0.*var'].constrain_fixed(1.)
m['.*ICM0.*W'].constrain_fixed(0)
m['.*ICM1.*var'].constrain_fixed(1.)
m['.*ICM1.*W'].constrain_fixed(0)

m.optimize()


# 创建预测点数列
def add_index_and_noise(X, po):
    l = po  # l对应着的是输出的维度，因此等于2
    index = np.zeros_like(X[:, 0].reshape(-1, 1))
    index1 = np.ones_like(X[:, 0].reshape(-1, 1))
    X_i = X
    for i in range(1, l):
        index_i = i * index1
        index = np.vstack([index, index_i])
        X = np.vstack([X, X_i])
    newX = np.hstack([X, index])
    noise_dict = {'output_index': newX[:, 4].astype(int)}
    return newX, noise_dict


newX, noise_dict = add_index_and_noise(X_test, po)
mean_all, var = m.predict(newX, Y_metadata=noise_dict)

split_array = np.array_split(mean_all, po)

import csv

outputFile = open('output_by_gpy.csv', 'w', newline='')
outputWriter = csv.writer(outputFile)
outputWriter.writerow(['name', 'value'])
# Qname=['Q-3/9','Q-3/10','Q0.5/9','Q0.5/10']
Qname = indices

for i, sub_array in enumerate(split_array):
    flat_list = np.array(split_array[i]).flatten().tolist()
    string = '[' + ' '.join(map(str, flat_list)) + ']'
    outputWriter.writerow([Qname[i] + '_gpy', string])
    a = y_test[:, i]
    outputWriter.writerow([Qname[i] + '_test', y_test[:, i]])

# outputWriter.writerow(['S11-9_gpy', split_array[0]])
# outputWriter.writerow(['S11-9_test', y_test[:,0]])
# outputWriter.writerow(['Q-3/9_gpy', split_array[1]])
# outputWriter.writerow(['Q-3/9_test', y_test[:,1]])
