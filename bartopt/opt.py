import numpy as np
from matplotlib import pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from joblib import load
m1 = load('BART-S11.joblib')
m2=load('BART-Q0.5-9.joblib')
# 这个预测计算过程非常的缓慢！！！

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([8,0.2,0.2,0.2]),
                         xu=np.array([15,3.35/3,1.575,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [my_y(x)]
        out["G"] = [(-1/2*x[0]+6*x[1]+4*x[2])/6]

def my_y(x):
    x=np.array(x).reshape(1,-1)
    pred_by_model1 = m1.predict(x)
    pred_by_model2 = m2.predict(x)
    pred_by_model2=-pred_by_model2
    y=np.array([pred_by_model1,pred_by_model2])
    # print(y)
    return y

problem = MyProblem()
# 下面是定义算法
from pymoo.algorithms.moo.nsga2 import NSGA2

algorithm = NSGA2(pop_size=300,
                  n_offsprings=15,
                  eliminate_duplicates=True)
#对于相对简单的问题，选择群体大小为40 (`pop_size=40`)，每代只有10个后代 (`n_offsprings=10)`。启用重复检查(`eliminate_duplicate =True`)，确保交配产生的后代与它们自己和现有种群的设计空间值不同。
# 终止标准，使用了一个相当小的40次迭代的算法

termination = get_termination("n_gen", 300)
# 优化
from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)

X = res.X #自变量坐标
F = res.F #因变量坐标
np.save('X200.npy', X)
np.save('F200.npy', F)
# print(F)

# 绘制散点图
plt.scatter(F[:,0], F[:,1])
# 添加标题和坐标轴标签
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.show()

