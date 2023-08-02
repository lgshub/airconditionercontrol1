import os
import numpy as np
import pandas as pd


file_list = ['绍_C_绍兴齐贤局_2F综合机房.csv','婺_A_云计算中心二号楼局_3F3-2电力电池室.csv','甬_C_余姚富巷母局_2F综合机房.csv']
file = file_list[2]
file_path = '../data2/202306120947/' + file
data = pd.read_csv(file_path)
cdict = {"return": [], "setting": [], "rack_avg": []}
for v in data.columns:
    if v[-2:]=='_r':
        cdict["return"].append(v)
    elif v[-2:]=='_s':
        cdict['setting'].append(v)
    elif v[-2:]=='_h':
        cdict['rack_avg'].append(v)
env_avg = data["env_avg"]
return_t = data[cdict["return"][-1]]
setting_t = data[cdict["setting"][-1]]
rack_avg = data[cdict["rack_avg"][-1]]
dataframe = data[["env_avg", cdict["return"][-1], cdict["rack_avg"][-1], cdict["setting"][-1]]]
# dataframe.to_csv(file, index=False)
xyz = dataframe.to_numpy(copy=True)
# print(np.sort(xyz))
ind = np.lexsort((xyz[:,0],xyz[:,1],xyz[:,2]))
# print(ind)
xyz2 = [xyz[i] for i in ind]
count = 0
max_count = 0
num_count = 0
for i, v in enumerate(xyz2):
    if (xyz2[i][:2] == xyz2[i-1][:2]).all() and (xyz2[i][2] != xyz2[i-1][2]).all():
        count += 1
        if max_count < count:
            max_count = count
    else:
        num_count += 1
        count = 0
print(max_count)
print('mean: ', len(xyz2)/num_count)

import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
#创建绘图区域
ax = plt.axes(projection='3d')
# ax.view_init(70,45)
ax.scatter3D(rack_avg, return_t, setting_t)
ax.set_title('3d Scatter plot')
plt.savefig('3d.png')
