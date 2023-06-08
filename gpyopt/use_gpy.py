import GPy
import numpy as np
import pandas as pd
path="C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\"
train_filename =path+ 'data/FE_all.csv'

all_data = pd.read_csv(train_filename)
indices = ['L', 'g', 's', 't', 'S11-9', 'Q/-3_9']
all_data = np.array(all_data.loc[:, indices])


from sklearn.model_selection import train_test_split
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_data[:,0:4], all_data[:,4:], test_size=0.2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(train_data, train_targets, test_size=13, random_state=0)

# test_data = pd.read_csv(test_filename)
# indices = ['L', 'g', 's', 't', 'S11-9', 'Q/-3_9']



# LCM 多核
def lcm():
    K1 = GPy.kern.Bias(4)
    K2 = GPy.kern.RBF(4)
    K3 = GPy.kern.Matern32(4)
    lcm = GPy.util.multioutput.LCM(input_dim=4,
                                   num_outputs=2,
                                   kernels_list=[K1, K2, K3],
                                   W_rank=2)
    return lcm


Y1 =y_train[:,0].reshape(-1,1)

Y2 =y_train[:,1].reshape(-1,1)

# ICM 单核
# K= GPy.util.multioutput.ICM(input_dim=p, num_outputs=po, kernel=Matern32(4), W_rank=2)


K = lcm()
# 模型构建与参数优化
m = GPy.models.GPCoregionalizedRegression([X_train, X_train], [Y1, Y2], kernel=K)
# m['.*Mat32.var'].constrain_fixed(1.)  # For this kernel, B.kappa encodes the variance now.
m['.*ICM.*var'].unconstrain()
m['.*ICM0.*var'].constrain_fixed(1.)
m['.*ICM0.*W'].constrain_fixed(0)
m['.*ICM1.*var'].constrain_fixed(1.)
m['.*ICM1.*W'].constrain_fixed(0)
m.optimize()


# 创建预测点数列
def add_index_and_noise(X):
    l = 2  # l对应着的是输出的维度，因此等于2
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


newX, noise_dict = add_index_and_noise(X_test)
mean_of_two, var = m.predict(newX, Y_metadata=noise_dict)
l1 = mean_of_two[0:len(X_test[:, 0])]
l2 = mean_of_two[len(X_test[:, 0]):]
mean = np.hstack([l1, l2])

# import csv
# outputFile = open('output_by_gpy.csv', 'w', newline='')
# outputWriter = csv.writer(outputFile)
# outputWriter.writerow(['name', 'value'])
# outputWriter.writerow(['S11-9_gpy', mean[:, 0]])
# outputWriter.writerow(['S11-9_test', y_test[:,0]])
# outputWriter.writerow(['Q-3/9_gpy', mean[:, 1]])
# outputWriter.writerow(['Q-3/9_test', y_test[:,1]])
