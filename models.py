from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import warnings
from bartpy.sklearnmodel import SklearnModel
import GPy
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
from joblib import load
from keras.models import load_model
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.visualization.scatter import Scatter


def split_train_and_test(X_indices, Y_indices):
    data_filename = 'data\\FE_all.csv'
    all_data = pd.read_csv(data_filename)
    X = np.array(all_data.loc[:, X_indices])
    Y = np.array(all_data.loc[:, Y_indices])
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def LstmFitAndPred(X_train, y_train, X_test):
    mean = X_train.mean(axis=0)
    X_train -= mean
    std = X_train.std(axis=0)
    X_train /= std  # 归一化处理
    X_test -= mean
    X_test /= std
    warnings.filterwarnings('ignore')
    # 模型搭建
    model = Sequential()  # 顺序模型，核心操作是添加layer（图层）
    model.add(LSTM(units=75, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=150, return_sequences=True))
    # model.add(LSTM(units=40),return_sequences=True, input_shape=(None,50))
    model.add(LSTM(units=25))
    # model.add(Dense(10))
    n = y_train.shape[1]
    model.add(Dense(n))  # 全连接层，输出层
    model.summary()
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # 选择优化器，并指定损失函数
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(X_train, y_train, epochs=4961, batch_size=1, verbose=2)
    model.fit(X_train, y_train, epochs=4961, batch_size=1, verbose=2)
    model.save('TrainedModels\\f9f10S11Q0.5-9.h5')
    y_t_lstm = model.predict(X_test)
    return y_t_lstm


def BARTFitAndPred(X_train, y_train, X_test):
    def get_clean_model():
        return SklearnModel(n_chains=5,
                            n_jobs=-1,
                            n_burn=200,
                            n_samples=1200,
                            n_trees=60,
                            initializer=None)

    for i in range(y_train.shape[1]):
        model = get_clean_model()
        model.fit(X_train, y_train[:, i])
        dump(model, 'TrainedModels/BART_' + Y_indices[i] + '.joblib')
        pred = model.predict(X_test).reshape(-1, 1)
        if i == 0:
            all_pred = pred
        else:
            all_pred = np.hstack([all_pred, pred])
    return all_pred


def GpyFitAndPred(X_train, y_train, X_test):
    # LCM 多核
    def lcm(X_train, y_train):
        n_in = X_train.shape[1]
        n_out = y_train.shape[1]
        K1 = GPy.kern.Bias(n_in)
        K2 = GPy.kern.RBF(n_in)
        K3 = GPy.kern.Matern32(n_in)
        lcm = GPy.util.multioutput.LCM(input_dim=n_in,
                                       num_outputs=n_out,
                                       kernels_list=[K1, K2, K3],
                                       W_rank=2)
        return lcm

    def XY(X_train, y_train, num_output):
        # 输出时预测点 X格式重构
        X = [X_train]
        Y = [y_train[:, 0].reshape(-1, 1)]
        k = num_output
        i = 2
        while i <= k:
            X.append(X_train)
            Y.append(y_train[:, i - 1].reshape(-1, 1))
            i += 1
        return X, Y

    po = y_train.shape[1]
    X, Y = XY(X_train, y_train, po)

    K = lcm(X_train, y_train)
    # 模型构建与参数优化
    m = GPy.models.GPCoregionalizedRegression(X, Y, kernel=K)
    # m['.*Mat32.var'].constrain_fixed(1.)  # For this kernel, B.kappa encodes the variance now.
    m['.*ICM.*var'].unconstrain()
    m['.*ICM0.*var'].constrain_fixed(1.)
    m['.*ICM0.*W'].constrain_fixed(0)
    m['.*ICM1.*var'].constrain_fixed(1.)
    m['.*ICM1.*W'].constrain_fixed(0)
    m.optimize()

    # 1: Saving a model:
    np.save('TrainedModels/MOGP.npy', m.param_array)  # 模型保存

    # 创建预测点数列
    def add_index_and_noise(X, y_test):
        l = y_test.shape[1]  # l对应着的是输出的维度
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

    newX, noise_dict = add_index_and_noise(X_test, y_test)
    mean, var = m.predict(newX, Y_metadata=noise_dict)
    a = np.array(mean).flatten()  # 二维数组扁平化
    b = np.array(var).flatten()
    split_array = np.array_split(a, po)  # 按照输出维度将一列的数据等分
    mean_value = np.transpose(np.array(split_array))
    split_array = np.array_split(b, po)  # 按照输出维度将一列的数据等分
    var_value = np.transpose(np.array(split_array))
    return mean_value, var_value


def SaveToExcel(data, headers, name):
    df = pd.DataFrame(data, columns=headers)
    # 定义要保存的Excel文件路径
    excel_file = 'Preds/' + name + '.xlsx'
    # 将DataFrame写入Excel文件
    df.to_excel(excel_file, index=False)


def PlotPred(target_y):
    data = pd.read_excel(target_y)
    test_data = pd.read_excel('data\y_test.xlsx')
    headers = data.columns
    n = data.shape[0]  # 样本点数量
    # 创建一个窗口，并设置子图布局为2行2列
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # 生成数据
    x = range(1, n + 1)
    data = data.values
    test_data = test_data.values
    # 在每个子图中绘制曲线
    axes[0, 0].plot(x, data[:, 0], label=headers[0], marker='o', color='blue')
    axes[0, 0].plot(x, test_data[:, 0], label='True', marker='s', color='red')
    axes[0, 0].set_title(headers[0])

    axes[0, 1].plot(x, data[:, 1], label=headers[1], marker='o', color='blue')
    axes[0, 1].plot(x, test_data[:, 1], label='True', marker='s', color='red')
    axes[0, 1].set_title(headers[1])

    axes[1, 0].plot(x, data[:, 2], label=headers[2], marker='o', color='blue')
    axes[1, 0].plot(x, test_data[:, 2], label='True', marker='s', color='red')
    axes[1, 0].set_title(headers[2])

    axes[1, 1].plot(x, data[:, 3], label=headers[3], marker='o', color='blue')
    axes[1, 1].plot(x, test_data[:, 3], label='True', marker='s', color='red')
    axes[1, 1].set_title(headers[3])

    # 设置图例
    for ax in axes.flat:
        ax.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()


def PlotErrors(target_y):
    data = pd.read_excel(target_y)
    test_data = pd.read_excel('data\y_test.xlsx')
    headers = data.columns
    n = data.shape[0]  # 样本点数量
    # 创建一个窗口，并设置子图布局为2行2列
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # 生成数据
    x = np.linspace(1, n, n)
    data = data.values
    test_data = test_data.values
    errors = np.abs((data - test_data) / test_data)
    # 在每个子图中绘制曲线
    axes[0, 0].plot(x, errors[:, 0], label='True', marker='s', color='red')
    axes[0, 0].set_title(headers[0])

    axes[0, 1].plot(x, errors[:, 1], label='True', marker='s', color='red')
    axes[0, 1].set_title(headers[1])

    axes[1, 0].plot(x, errors[:, 2], label='True', marker='s', color='red')
    axes[1, 0].set_title(headers[2])

    axes[1, 1].plot(x, errors[:, 3], label='True', marker='s', color='red')
    axes[1, 1].set_title(headers[3])

    # 设置图例
    for ax in axes.flat:
        ax.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()


def LoadModel(model, X_train, y_train):
    if model == 'MOGP':
        def lcm(X_train, y_train):
            n_in = X_train.shape[1]
            n_out = y_train.shape[1]
            K1 = GPy.kern.Bias(n_in)
            K2 = GPy.kern.RBF(n_in)
            K3 = GPy.kern.Matern32(n_in)
            lcm = GPy.util.multioutput.LCM(input_dim=n_in,
                                           num_outputs=n_out,
                                           kernels_list=[K1, K2, K3],
                                           W_rank=2)
            return lcm

        K = lcm(X_train, y_train)

        def XY(X_train, y_train, num_output):
            # 输出时预测点 X格式重构
            X = [X_train]
            Y = [y_train[:, 0].reshape(-1, 1)]
            k = num_output
            i = 2
            while i <= k:
                X.append(X_train)
                Y.append(y_train[:, i - 1].reshape(-1, 1))
                i += 1
            return X, Y

        po = y_train.shape[1]
        X, Y = XY(X_train, y_train, po)
        m_load = GPy.models.GPCoregionalizedRegression(X, Y, kernel=K, initialize=False)
        m_load.update_model(False)  # do not call the underlying expensive algebra on load
        m_load.initialize_parameter()  # Initialize the parameters (connect the parameters up)
        m_load[:] = np.load('TrainedModels/MOGP.npy')  # Load the parameters
        m_load.update_model(True)  # Call the algebra only once
        return m_load
    if model == 'BART':
        m1 = load('TrainedModels/BART_f9.joblib')
        m2 = load('TrainedModels/BART_f10.joblib')
        m3 = load('TrainedModels/BART_S11-9.joblib')
        m4 = load('TrainedModels/BART_Q0.5_9.joblib')
        return m1, m2, m3, m4
    if model == 'LSTM':
        model = load_model('TrainedModels/f9f10S11Q0.5-9.h5')
        return model


def add_index_and_noise(X, p):
    l = p  # l对应着的是输出的维度
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


def GPoutputsTrans(a, po):
    for i in range(a.shape[1]):
        split_array = np.array_split(a[:, i], po)  # 按照输出维度将一列的数据等分
        value_array = np.transpose(np.array(split_array))
    return value_array


def PlotScatter(dataname):
    F = np.load('Opt/' + dataname + '_F.npy')
    fl = F.min(axis=0)
    fu = F.max(axis=0)
    print(f"Scale f1: [{fl[0]}, {fu[0]}]")
    print(f"Scale f2: [{fl[1]}, {fu[1]}]")
    print(f"Scale f3: [{fl[2]}, {fu[2]}]")

    weights = np.array([0.5, 0.3, 0.2])
    i = PseudoWeights(weights).do(F)
    print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))

    # plt.figure(figsize=(7, 5))
    # plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    # plt.scatter(F[i, 0], F[i, 1], F[i, 2],marker="x", color="red", s=200)
    # plt.title("Pareto Front")
    # plt.show()
    plot = Scatter()
    plot.add(F)
    plot.axis_labels=['df','S11','Q']
    plot.show()


def chooseModel(choose='LSTM'):
    # 将模型输出到当前位置空间
    if choose == 'BART':
        m1, m2, m3, m4 = LoadModel('BART', X_train, y_train)
        m = [m1, m2, m3, m4]
        return m
    if choose == 'LSTM':
        m = LoadModel('LSTM', X_train, y_train)
        return m
    if choose == 'MOGP':
        m = LoadModel('MOGP', X_train, y_train)
        return m


def ParetoOpt():
    problem = MyProblem()
    algorithm = NSGA2(pop_size=200,
                      n_offsprings=10,
                      eliminate_duplicates=True)
    # 选择群体大小为40 (`pop_size=40`)，每代只有10个后代 (`n_offsprings=10)`。启用重复检查(`eliminate_duplicate
    # =True`)，确保交配产生的后代与它们自己和现有种群的设计空间值不同。 终止标准
    termination = get_termination("n_gen", 150)
    # 优化
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True)
    X = res.X
    F = res.F
    return X, F

def my_y(x, choose):
    x = np.array(x).reshape(1, -1)
    if choose == 'BART':
        [m1, m2, m3, m4] = m
        y1 = m[0].predict(x)
        y2 = m[1].predict(x)
        y12 = -np.abs(y1 - y2)  # 作为优化目标【最小化】，要调整为负数
        y3 = m[2].predict(x)
        y4 = m[3].predict(x)
        y = np.array([y12, y3, y4])
        # print(y)
    if choose == 'LSTM':
        # m = LoadModel('LSTM', X_train, y_train)
        yl = -m.predict(x, verbose=0)
        y = np.hstack([-np.abs(yl[0][1] - yl[0][0]), -yl[0][2], yl[0][3]])
    if choose == 'MOGP':
        # m = LoadModel('MOGP', X_train, y_train)
        p = 4
        X, noise = add_index_and_noise(x, p)
        a, b = m.predict(X, Y_metadata=noise)
        # y = GPoutputsTrans(a, 4)
        y = np.array([-np.abs(a[0] - a[1]), a[2], -a[3]])
    return y

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=3,
                         n_constr=1,
                         xl=np.array([8, 0.2, 0.2, 0.2]),
                         xu=np.array([15, 3.35 / 3, 1.575, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [my_y(x, choose)]
        out["G"] = [(-1 / 2 * x[0] + 6 * x[1] + 4 * x[2]) / 6]


X_indices = ['L', 'g', 't', 's']
Y_indices = ['f9', 'f10', 'S11-9', 'Q0.5_9']
X_train, X_test, y_train, y_test = split_train_and_test(X_indices, Y_indices)


# 模型训练
# a=BARTFitAndPred(X_train, y_train, X_test)
# a=LstmFitAndPred(X_train, y_train, X_test)
# a,b=GpyFitAndPred(X_train, y_train, X_test)
# SaveToExcel(a,Y_indices,'MOGP')
# SaveToExcel(b,Y_indices,'MOGP_var')

# 绘制预测结果和误差
# excelname='LSTM'
# PlotPred('Preds/'+excelname+'.xlsx')
# PlotPred('Preds/MOGP.xlsx')
# PlotErrors('Preds/MOGP.xlsx')


# 模型加载
# model=LoadModel('BART',X_train,y_train)

# p=m4.predict(X_test)
# print(p)
# X,noise=add_index_and_noise(X_test,y_test)
# a,b=m.predict(X, Y_metadata=noise)
# a=GPoutputsTrans(a,4)


# BART模型的y设置





# choose = 'MOGP'
choose = 'BART'
# choose = 'LSTM'
m = chooseModel(choose)
#
X, F = ParetoOpt()
np.save('Opt/' +choose+'_X.npy', X)
np.save('Opt/' +choose+'_F.npy', F)
# print(F)
PlotScatter(choose)
