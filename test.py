import numpy as np
import pandas as pd
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem

from pymoo.util.ref_dirs import get_reference_directions
a=np.load ('Opt/MOGP_F.npy')
am=np.load ('Opt/LSTM_F.npy')
# b=pd.read_excel('Preds/MOGP.xlsx')
print(a)
print(am)
