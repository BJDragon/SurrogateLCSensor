import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
path="C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\LSTMopt\\"
# path="C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\bartopt\\"
# x = np.load(path+'X.npy')
F = np.load(path+'F.npy')
# F[:,1]=-F[:,1]
# # 绘制散点图
# plt.scatter(y[:,0], y[:,1])
#
# # 添加标题和坐标轴标签
# plt.title('Scatter Plot')
# plt.xlabel('X')
# plt.ylabel('Y')
#
# # 显示图形
# plt.show()
#1.目标函数归一化
fl = F.min(axis=0)
fu = F.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

# 1.归一化法
# approx_ideal = F.min(axis=0)
# approx_nadir = F.max(axis=0)
# plt.figure(figsize=(7, 5))
# plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.scatter(approx_ideal[0],approx_ideal[1],facecolors='none',edgecolors='red',marker="*", s=100, label="Ideal Point (Approx)")
# plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none', edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
# plt.title("Objective Space")
# plt.legend()
# plt.show()
#
# nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
# fl = nF.min(axis=0)
# fu = nF.max(axis=0)
# print(f"Scale f1: [{fl[0]}, {fu[0]}]")
# print(f"Scale f2: [{fl[1]}, {fu[1]}]")
# plt.figure(figsize=(7, 5))
# plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.title("Objective Space")
# plt.show()
# # Scale f1: [0.0, 1.0] # Scale f2: [0.0, 1.0]


# 3.伪权重
from pymoo.mcdm.pseudo_weights import PseudoWeights
weights=np.array([0.5, 0.5])
i = PseudoWeights(weights).do(F)
print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Pareto Front")
plt.show()
# Best regarding Pseudo Weights: Point
# i = 39
# F = [58.52211061 0.06005482]
