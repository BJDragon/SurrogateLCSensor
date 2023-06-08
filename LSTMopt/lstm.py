import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.saving.save import load_model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Activation
# from tensorflow.keras.layers import Conv1D, MaxPooling1D
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import warnings

warnings.filterwarnings('ignore')

import functions
data=functions.Data_preprocess('C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\data/FE_all.csv',0)
# a=data.atest('sss')
X_indices=['L','g','t','s']
Y_indices= ['S11-9','Q/0.5_9']
X_train, X_test, y_train, y_test=data.split_train_and_test(X_indices,Y_indices)
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std # 归一化处理？
#
# # 测试数据必须以训练数据的均值和标准差进行变换，这样两者的变换才一致
X_test -= mean
X_test /= std

# 模型搭建
model = Sequential()  # 顺序模型，核心操作是添加layer（图层）
model.add(LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=100, return_sequences=True))
# model.add(LSTM(units=40),return_sequences=True, input_shape=(None,50))
model.add(LSTM(units=50))
# model.add(Dense(10))
n=y_train.shape[1]
model.add(Dense(n))  # 全连接层，输出层
model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # 选择优化器，并指定损失函数
model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train, y_train, epochs=4961, batch_size=1, verbose=2)
model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=2)# 近3000次训练即可将误差缩小到个位数，从上万的误差开始缩减

# 数据可视化
# result = model.predict(test_data[:])

# model.save('lstm_models\\S11-Q0.5-9.h5')
# model=load_model('S11-Q_0.5/S11-Q0.5-9.h5')
y_t_lstm = model.predict(X_test) # 测试点
#print(y_t_lstm)


# 假设您有一个名为data的NumPy数组变量
# 将数组转换为DataFrame
# headers = ['S11/9','Q0.5/9']
# df = pd.DataFrame(y_t_lstm,columns=headers)

# 定义要保存的Excel文件路径
# excel_file = 'lstm_pred.xlsx'

# 将DataFrame写入Excel文件
# df.to_excel(excel_file, index=False)

# a=functions.SaveData(y_t_lstm, Y_indices, 'S11-Q0.5.csv')
# a.save_it()
# b=functions.SaveData(y_test,Y_indices,'responsed_y.csv')
# b.save_it()

# 加载模型
# from keras.models import load_model
# model = load_model('f9-S11-Q-3-Q0.5.h5')
# a=functions.SaveData(y_t_lstm,Y_indices,'output_by_lstm.csv')
# ml=a.modifyIndices('_lstm')
# a.save_it()
# mt=a.modifyIndices("_test")
# a.add_data_to_csv(y_test,mt)
#
# b=functions.Plot('testopt.csv')
# ind1=['f9_lstm','f9_test']
# ind2=['Q/-3_9_lstm','Q/-3_9_test']
# b.plot_two_lines(ind1,ind2)








