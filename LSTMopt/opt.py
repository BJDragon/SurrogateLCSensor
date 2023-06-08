import numpy as np
from matplotlib import pyplot as plt
from pymoo.core.problem import ElementwiseProblem
#加载神经网络的参数
from keras.models import load_model
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination import get_termination
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
path="C:\\Users\\22965\\OneDrive - mail.ecust.edu.cn\\桌面\\Python Code\\2.Application\\lc\\"

model=load_model(path+'lstm_models/S11-Q0.5-9.h5')
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
    y=model.predict(x, verbose=0)
    y[0][1]=-y[0][1]
    # print(y)
    return y

problem = MyProblem()
# 下面是定义算法
from pymoo.algorithms.moo.nsga2 import NSGA2

algorithm = NSGA2(pop_size=200,
                  n_offsprings=30,

                  eliminate_duplicates=True)
#对于相对简单的问题，选择群体大小为40 (`pop_size=40`)，每代只有10个后代 (`n_offsprings=10)`。启用重复检查(`eliminate_duplicate =True`)，确保交配产生的后代与它们自己和现有种群的设计空间值不同。
# 终止标准，使用了一个相当小的40次迭代的算法

termination = get_termination("n_gen", 200)
# 优化
from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)

X = res.X #自变量坐标
F = res.F #因变量坐标
np.save('X.npy', X)
np.save('F.npy', F)
print(F)

