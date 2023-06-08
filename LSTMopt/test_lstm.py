from matplotlib import pyplot as plt
import functions
# 加载模型
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import warnings

warnings.filterwarnings('ignore')
import functions

data = functions.Data_preprocess('data/FE_all.csv', 0)
X_indices = ['L', 'g', 't', 's']
Y_indices = ['f9', 'S11-9', 'Q/-3_9', 'Q/0.5_9']
X_train, X_test, y_train, y_test = data.split_train_and_test(X_indices, Y_indices)
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std  # 归一化处理？
# # 测试数据必须以训练数据的均值和标准差进行变换，这样两者的变换才一致
X_test -= mean
X_test /= std
model = load_model('lstm_models/f9-S11-Q-3-Q0.5.h5')
y_t_lstm = model.predict(X_test)
# a=functions.SaveData(y_t_lstm,Y_indices,'output_by_lstm.csv')
# ml=a.modifyIndices('_lstm')
# a.save_it()
# mt=a.modifyIndices("_test")
# a.add_data_to_csv(y_test,mt)

b = functions.Plot('output_by_lstm.csv', Y_indices)
ind1 = Y_indices
ind2 = Y_indices
# b.plot_two_lines(ind1,ind2,Y_indices)


import pandas as pd

file1 = 'f9-S11-Q-3-Q0.5.csv'
file2 = 'responsed_y.csv'
j = 1
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
for i in Y_indices:
    data_lstm = pd.read_csv(file1, usecols=[i])
    data_test = pd.read_csv(file2, usecols=[i])
    # plt.subplot(4, 1, j)
    axs[j-1,0].plot(data_lstm, label='lstm' + Y_indices[j - 1])
    axs[j-1,0].plot(data_test, label='testopt' + Y_indices[j - 1], color='blue')
    plt.xlabel('n')
    plt.title('No.' + str(j))
    plt.legend()

    # plt.subplot(4, 2, j)
    axs[j-1,1].plot((data_lstm-data_test)/data_test, label='error: ' + Y_indices[j - 1],color='red')
    plt.xlabel('n')
    plt.title('No.' + str(j))
    plt.legend()
    j += 1
# 调整图框布局
plt.tight_layout()
plt.show()
