import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = np.load('wy.npy', allow_pickle=True)
# keys = file[0].keys()
# print(keys.sort())
print(file)