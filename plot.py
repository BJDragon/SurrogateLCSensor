import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from joblib import load

# LSTM模型
path = "C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\"
# model = load_model(path + 'lstm_models/S11-Q0.5-9.h5')

# BART模型
m1 = load(path + 'bartopt\\' + 'BART-S11.joblib')
m2 = load(path + 'bartopt\\' + 'BART-Q0.5-9.joblib')

# data = functions.Data_preprocess(
#     'C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\data/FE_all.csv', 0)
# y_test = pd.read_excel('y_test.xlsx')
# print(y_test)
# y_model = pd.read_excel('LSTMopt\lstm_pred.xlsx')
# print(y_model)
X=np.load(path+'data\\'+'X_train.npy')
y_model=np.hstack([m1.predict(X).reshape(-1,1),m2.predict(X).reshape(-1,1)])
y_test=np.load(path+'data\\'+'y_train.npy')


# column_names = y_test.columns
column_names =['S11','Q']
titles = ['lstm_S11_9.jpg', 'lstm_Q0.5_9.jpg']



for i in range(0, 2):
    # print(i)
    plt.figure(i + 1)
    plt.subplot(2, 1, 1)
    # 绘制第一条折线
    # plt.plot(y_test[column_names[i]], label='true', marker='o')
    plt.plot(y_test[:,i], label='true', marker='o')
    # 绘制第二条折线
    plt.plot(y_model[:,i], label='bart', marker='s')
    # 添加图例
    plt.legend()
    # 添加标题和坐标轴标签
    plt.title(column_names[i])
    plt.xlabel('samples')
    # plt.ylabel('S11')

    plt.subplot(2, 1, 2)
    # 绘制第一条折线
    plt.plot((y_model[:,i] - y_test[:,i]) / y_test[:,i], label='errors', marker='o',
             color='red')
    # 添加图例
    plt.legend()
    # 添加标题和坐标轴标签
    plt.title('error')
    plt.xlabel('samples')
    # 调整图框布局
    plt.tight_layout()
    plt.savefig(titles[i])
# 显示图形
plt.show()
# plt.figure(1);plt.savefig(titles[i])
# plt.figure(2);plt.savefig('bart_Q0.5_9.jpg')
