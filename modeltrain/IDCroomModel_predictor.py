import os
import joblib
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
# from pylab import *
import logging
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, accuracy_score


sys.path.append('../')
# from config.constants import CONVERTERS

logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s')


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield root + '/' + f

CONVERTERS = {'room_id': str, 'acu_id': str, 'rack_id': str, 'setting_t': float, 'setting_h': float, 'cp_status': float,
        'fd_status': float, 'fd_degree': float, 
        'temp': float, 'fan_speed': float, 'return_t': float, 'acu_power': float, 'outdoor_t': float,
        'outdoor_h': float}

# idxs = ['rack_h', 'rack_l', 'rack_avg', 'env_avg', 'env_h', 'temp', 'return_t']
# idxs = ['env_avg', 'env_h', 'return_t']
idxs = ['rack_h', 'rack_l', 'rack_avg', 'env_avg', 'env_h', 'return_t']

def _label(data, choice=-1):
    idx = data.columns
    conditioner={'setting_t': [], 'return_t': [], 'power': [], 'rack_avg': []}
    # for a in idx:
    #     if '_s' in a:
    #         conditioner['setting_t'].append(data[a])        
    #     elif '_r' in a:
    #         conditioner['return_t'].append(data[a])
    #     # elif '_p' in a:
    #     #     conditioner['power'].append(data[a])
    #     # elif '_h' in a:
    #     #     conditioner['rack_area'].append(data[a])
    # # data['temp'] = data['env_avg']
    # data['setting_t'] = conditioner['setting_t'][choice]
    # data['return_t'] = conditioner['return_t'][choice]
    # data['rack_avg'] = conditioner['rack_avg'][choice]
    # return data[idxs + ['setting_t']]
    return data[idxs + ['low', 'high', '控制范围', '环境最高温', '机柜最高温', '机柜预警线' , '机柜报警线', '回风预警线', \
        '环境报警线'] + ['setting_t']]

def main():
    file = ['绍_C_绍兴齐贤局_2F综合机房.csv','婺_A_云计算中心二号楼局_3F3-2电力电池室.csv', \
    '甬_C_余姚富巷母局_2F综合机房.csv', '禾_C_海宁干河街局_4F政府云机房.csv', \
        '禾_D_海盐武原城南局_2F综合机房.csv']
    # path = '../data/data2/202306120947/' + file[2]
    # data = pd.read_csv(path)
    # testfile = '../data/train_data/待打标数据汉口新区金银湖IDC_12机房202206010614.csv'
    # testfile = '../data/data2/202306120947/' + file[4]
    testfile = 'base.csv'
    rstfile = testfile.replace(".csv", "_pred.csv")
    # '../data/test_data/待打标数据武清IDC2_9机房04070409_pred.csv'
    data = pd.read_csv(testfile)
    logging.info(data[:2])
    data = _label(data, choice=0)
    # data['return_t'] = data.loc[:, data.columns[-1]]
    # data_x = data[idxs + ['环境最高温', '机柜最高温', '机柜预警线' , '机柜报警线', '回风预警线', \
    #     '环境报警线']]
    data_x = data[idxs + ['low', 'high', '环境最高温', '机柜最高温', '机柜预警线' , '机柜报警线', '回风预警线', \
        '环境报警线']]
    
    
    data_y = data['setting_t'].apply(str)
    # # y = 'fan_rated_power','fan_min_power','fd_degree_set's
    # data_x.loc[:, ['outdoor_t']] = data_x.loc[:, ['outdoor_t']].astype('float16')
    # data_x.loc[:, ['outdoor_h']] = data_x.loc[:, ['outdoor_h']].astype('float16')
    lgb_load = joblib.load('../model/AC_Model_HB2.pkl')

    logging.info(data[:2])
    pre_y = lgb_load.predict(data_x, num_iteration=lgb_load.best_iteration_)
    data['pre_y'] = pre_y
    logging.info(len(data))  # df.groupby(['A','B']).B.agg('count').to_frame('c').reset_index()
    
    accuracy = accuracy_score(data_y, pre_y)
    logging.info("Test Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # data['count'] = data.groupby(['room_id', 'acu_id', 'pre_y'])['pre_y'].transform(
    #     'count')

    data.to_csv(rstfile, encoding='gbk')


if __name__ == '__main__':
    main()
