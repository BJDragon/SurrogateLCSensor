import pandas as pd
import numpy as np
# path="C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\"
path="C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\LSTMopt\\"
x = np.load(path+'X.npy')
y = np.load(path+'F.npy')
data=np.hstack((x,y))
# 假设您有一个名为data的NumPy数组变量
# 将数组转换为DataFrame
headers = ['L', 'g', 't','s','S11/9','Q0.5/9']
df = pd.DataFrame(data,columns=headers)

# 定义要保存的Excel文件路径
excel_file = '../LSTMopt/lstmopt.xlsx'

# 将DataFrame写入Excel文件
df.to_excel(excel_file, index=False)
