import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

idxs = ['rack_h', 'rack_l', 'rack_avg', 'env_avg', 'env_h']
            
def lasting(x):
    L = 1
    last = -1
    ls = []
    # dx = (np.array([x, 0]) - np.array([0, x]))[:-2]
    for v in x:
        L = (v==last)*L + 1
        ls.append(L)
        last = v
    return np.max(ls)

file_path = '../data2/202306120947/'
_list = os.listdir(file_path)
f_list = {}
# print(_list)

#  save+ 里存的是人工筛选出来的一些机房
saved = pd.read_csv('save+.csv').to_numpy()


for f in _list:
    data = pd.read_csv(file_path + f)
    sets = []
    flag = 0
    for v in data.columns:
        if v[-2:] == '_s':
            temp = data[v].to_numpy()*1.0
            nans = np.isnan(temp)
            temp = np.delete(temp, np.where(nans))
            if len(temp) < 1e3:
                continue
            dT = max(temp) - min(temp)
            lst = lasting(temp)
            if lst < 1e10 and np.all([u in data.columns for u in idxs]) and f[:-4] in saved:
                flag = 1
                f_list[f[:-4]] = lst
print(f_list)
np.save('wy', f_list)
_save = pd.DataFrame(f_list.keys())
_save.to_csv('save.csv')