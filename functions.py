"""
要有哪几个部分？
    1、数据的导入
    2、数据输入模型并完成拟合和预测
    3、数据的后处理
"""
import numpy as np
import pandas as pd
import os
# import GPy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Data_preprocess(object):
    """
    用于处理excel表的数据
    """

    def __init__(self,
                 data_filename: str,
                 mul_sheets: bool):
        self.data_filename = data_filename
        self.mul_sheets = mul_sheets
        self.all_data = pd.read_csv(self.data_filename)
        self.filename = os.path.basename(self.data_filename)
        self.path = os.path.dirname(self.data_filename)

    def check_file_existence(self):
        # 检查是都存在指定的文件
        directory = self.path
        sublist = self.filename
        for filename in sublist:
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath) and os.path.isfile(filepath):
                # 如果存在这个文件命名的文件存在，则不做处理
                pass
            else:
                # 该文件存在，但是后缀不对时另存文件
                basename = os.path.splitext(filename)[0]  # 获取文件名（不包含扩展名）
                for file in os.listdir(directory):
                    if file.startswith(basename) and os.path.isfile(os.path.join(directory, file)):
                        Data_preprocess.convert_xlsx_to_csv()

        return None

    def convert_xlsx_to_csv(self):
        # 将指定的文件转换成csv格式，当前仅能使用单独的表格，不适用于多工作表
        input_folder = self.path
        output_folder = self.path
        # 检查输出文件夹是否存在，如果不存在则创建它
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历指定文件夹中的所有文件
        for filename in os.listdir(input_folder):
            if filename.endswith(".xlsx"):
                # 构建输入文件的完整路径
                input_path = os.path.join(input_folder, filename)

                # 读取.xlsx文件中的所有工作表
                xl = pd.ExcelFile(input_path)
                sheet_names = xl.sheet_names

                # 遍历每个工作表，将其另存为独立的.csv文件
                for sheet_name in sheet_names:
                    # 读取当前工作表的数据
                    df = xl.parse(sheet_name)

                    # 构建输出文件的完整路径
                    output_filename = f"{os.path.splitext(filename)[0]}_{sheet_name}.csv"
                    output_path = os.path.join(output_folder, output_filename)

                    # 将数据保存为.csv文件
                    df.to_csv(output_path, index=False)

    def split_train_and_test(self, X_indices, Y_indices):
        X = np.array(self.all_data.loc[:, X_indices])
        Y = np.array(self.all_data.loc[:, Y_indices])
        # 将数据集分为训练集和测试集
        # X_train, X_test, y_train, y_test = train_test_split(train_data, train_targets, test_size=13, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
        p = X_train.shape[1]
        n = X_train.shape[0]
        po = y_train.shape[1]
        no = y_train.shape[0]
        return X_train, X_test, y_train, y_test


class SaveData():
    def __init__(self,
                 data,
                 indices,
                 filename):
        self.data = data
        self.indices = indices
        self.filename = filename

    def save_it(self):
        try:
            ind = self.modified_indices
        except:
            ind = self.indices
        df = pd.DataFrame(self.data, columns=ind)
        df.to_csv(self.filename, mode='w', index=False)

    def modifyIndices(self, added_str):
        self.modified_indices = [item + added_str for item in self.indices]
        return self.modified_indices

    def add_data_to_csv(self, data, indices):
        df = pd.DataFrame(data, columns=indices)  # columns=是不可缺少的
        df_existing = pd.read_csv(self.filename)
        df_combined = pd.concat([df_existing, df], axis=1)
        df_combined.to_csv(self.filename, index=False)


class Plot():
    """
    aims:给定一个包含所有数据的文件，针对性读取并做出所需要的图形
    """

    def __init__(self,
                 file,
                 indices
                 ):
        self.file = file
        self.indices=indices

    def readcsv(self, indices):
        data = pd.read_csv(self.file)
        readed_data = data.loc(indices)
        return readed_data

    def plot_two_lines(self, indices1, indices2, titles, compare=0):
        # data = pd.read_csv(self.file)
        # pd.read_csv('data.csv', usecols=['列1', '列3'])
        data1 = pd.read_csv(self.file, usecols=indices1)
        data2 = pd.read_csv(self.file, usecols=indices2)

        # 创建第一个图框
        plt.subplot(2, 1, 1)  # 2行1列，选择第1个子图
        plt.plot(data1[indices1[0]], label=indices1[0])
        plt.plot(data1[indices1[1]], label=indices1[1])
        plt.xlabel('n')
        # plt.ylabel('Q3_9')

        plt.title('No.1')
        plt.legend()

        # 创建第二个图框
        plt.subplot(2, 1, 2)  # 2行1列，选择第2个子图
        plt.plot(data2[indices2[0]], label=indices2[0])
        plt.plot(data2[indices2[1]], label=indices2[1])
        plt.xlabel('n')
        # plt.ylabel('Q3_10')
        plt.title('No.2')
        plt.legend()
        # 调整图框布局
        plt.tight_layout()
        # 显示图形
        plt.show()



# class MOGP_LMC(object):
#     """
#     用于高斯多输出回归过程中的设置
#     包括：
#         核函数的设置
#         参数的拟合
#         预测处理
#     """
#
#     def lcm(input_dim=4, num_outputs=4):
#         p = input_dim
#         K1 = GPy.kern.Bias(p)
#         K2 = GPy.kern.RBF(p)
#         K3 = GPy.kern.Matern32(p)
#         lcm = GPy.util.multioutput.LCM(input_dim,
#                                        num_outputs,
#                                        kernels_list=[K1, K2, K3],
#                                        W_rank=2)
#         return lcm
#
#     def XY(X_train, y_train, num_output):
#         X = [X_train]
#         Y = [y_train[:, 0].reshape(-1, 1)]
#         k = num_output
#         i = 2
#         while i <= k:
#             X.append(X_train)
#             Y.append(y_train[:, i - 1].reshape(-1, 1))
#             i += 1
#         return X, Y
#
#     X, Y = XY(X_train, y_train, po)
#
#     # K = icm()
#     K = lcm(p, po)
#     # 模型构建与参数优化
#     m = GPy.models.GPCoregionalizedRegression(X, Y, K)
#     # m = GPy.models.GPCoregionalizedRegression([X_train, X_train], [Y1, Y2], kernel=K)
#     # m['.*Mat32.var'].constrain_fixed(1.)  # For this kernel, B.kappa encodes the variance now.
#     m['.*ICM.*var'].unconstrain()
#     m['.*ICM0.*var'].constrain_fixed(1.)
#     m['.*ICM0.*W'].constrain_fixed(0)
#     m['.*ICM1.*var'].constrain_fixed(1.)
#     m['.*ICM1.*W'].constrain_fixed(0)
#     m.optimize()
