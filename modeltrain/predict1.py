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

def main():
    testfile = '../data/train_data/待打标数据汉口新区金银湖IDC_12机房202206010614.csv'
    rstfile = testfile.replace(".csv", "_pred.csv")
    # '../data/test_data/待打标数据武清IDC2_9机房04070409_pred.csv'
    data = pd.read_csv(testfile, converters=CONVERTERS)
    logging.info(data[:2])
    data_x = data[['return_t']]
    data_y = data['setting_t'].apply(str)
    # # y = 'fan_rated_power','fan_min_power','fd_degree_set's
    # data_x.loc[:, ['outdoor_t']] = data_x.loc[:, ['outdoor_t']].astype('float16')
    # data_x.loc[:, ['outdoor_h']] = data_x.loc[:, ['outdoor_h']].astype('float16')
    lgb_load = joblib.load('../model/AC_Model_HB2.pkl')

    logging.info(data[:2])
    pre_y = lgb_load.predict(data_x, num_iteration=lgb_load.best_iteration_)
    logging.info(len(data))  # df.groupby(['A','B']).B.agg('count').to_frame('c').reset_index()
    

if __name__ == '__main__':
    main()
