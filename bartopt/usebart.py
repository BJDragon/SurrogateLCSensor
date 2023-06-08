import pandas as pd
import numpy as np
from bartpy.sklearnmodel import SklearnModel
import functions

data = functions.Data_preprocess('C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\data/FE_all.csv', 0)
# a=data.atest('sss')
X_indices = ['L', 'g', 't', 's']
Y_indices = ['S11-9', 'Q0.5_9']
X_train, X_test, y_train, y_test = data.split_train_and_test(X_indices, Y_indices)
# 保存一下数据分组
# np.save('../data/X_train.npy', X_train)
# np.save('../data/y_train.npy', y_train)
# np.save('../data/X_test.npy', X_test)
# np.save('../data/y_test.npy', y_test)
#
# mean = X_train.mean(axis=0)
# X_train -= mean
# std = X_train.std(axis=0)
# X_train /= std
# X_test -= mean
# X_test /= std
#
# np.save('X_train'+'_norm'+'.npy', X_train)
# np.save('y_train'+'_norm'+'.npy', y_train)
# np.save('X_test'+'_norm'+'.npy', X_test)
# np.save('y_test'+'_norm'+'.npy', y_test)

from joblib import dump
# dump(model, 'regression_tree_model.joblib')
from joblib import load
# loaded_model = load('regression_tree_model.joblib')


def get_clean_model():
    return SklearnModel(n_chains=4,
                        n_jobs=50,
                        n_burn=200,
                        n_samples=1000,
                        n_trees=50,
                        initializer=None)


model1 = get_clean_model()
model1.fit(X_train, y_train[:, 0])
# dump(model1, 'BART-S11.joblib')
pred_by_model1 = model1.predict(X_test)

print()
#
# model2 = get_clean_model()
# model2.fit(X_train, y_train[:, 1])
# dump(model2, 'BART-Q0.5-9.joblib')
# pred_by_model2 = model2.predict(X_test)
#
# np.save('m1.npy', pred_by_model1)
# np.save('m2.npy', pred_by_model2)


# m1 = load('BART-S11.joblib')
# pred_by_model1=m1.predict(X_test)
# m2=load('BART-Q0.5-9.joblib')
# pred_by_model2=m2.predict(X_test)
# data=np.hstack((pred_by_model1.reshape(-1,1),pred_by_model2.reshape(-1,1)))
# # 假设您有一个名为data的NumPy数组变量
# # 将数组转换为DataFrame
# headers = ['S11/9','Q0.5/9']
# df = pd.DataFrame(data,columns=headers)
#
# # 定义要保存的Excel文件路径
# excel_file = 'bart_pred.xlsx'
#
# # 将DataFrame写入Excel文件
# df.to_excel(excel_file, index=False)

# import csv
#
# outputFile = open('bartopt\\output_by_bart.csv', 'w', newline='')
# outputWriter = csv.writer(outputFile)
# outputWriter.writerow(['name', 'value'])
# outputWriter.writerow(['S11-9_bart', pred_by_model])
# outputWriter.writerow(['S11-9_test', y_test[:, 0]])
#
# model = get_clean_model()
# model.fit(X_train, y_train[:, 1])
# pred_by_model = model.predict(X_test)
# outputWriter.writerow(['Q-3/9_bart', pred_by_model])
# outputWriter.writerow(['Q-3/9_test', y_test[:, 1]])
