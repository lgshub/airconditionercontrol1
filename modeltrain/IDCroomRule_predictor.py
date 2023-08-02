import numpy as np
import pandas as pd
import json
from lightgbm import LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, accuracy_score
from pylab import *
import joblib
import logging

logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s')


# data.columns is  ['setting_t', 'setting_h', 'cp_status','fan_speed',
# 'return_t', 'fd_degree', 'outdoor_t','outdoor_h', 'temp', 'activePower','kt_power',
# 'fan_rated_power', 'fan_min_power', 'fd_degree_set']

def air_conditioner_adjust(data, hotpoint, coolpoint):  # 空调参数调整函数
    result, sum_temps, count = {}, 0, 0
    # print(data)
    if data['cp_status'] == 0:
        data['fd_degree_set'], data['fan_rated_power'], data['fan_min_power'] = 0, 0, 0
    result['air_conditioner_id'] = data['air_conditioner_id']  # 空调ID
    result['fd_degree_set'] = data['fd_degree_set']  # 水阀开度
    result['fan_rated_power'] = data['fan_rated_power']  # 风机额度功率
    result['fan_min_power'] = data['fan_min_power']  # 风机最小功率
    result['rule'] = 'unchanged'

    for d in json.loads(data['racks']):  # 对空调对应的所有机柜进行遍历
        # rankmaxtemp = max([e["temp"] for e in d["temps"]])
        rankmaxtemp = d
        sum_temps += float(rankmaxtemp)
        count += 1
        if rankmaxtemp > hotpoint:  # 如果存在热点
            result['fd_degree_set'] = 80  # min(100, 80)
            result['fan_rated_power'] = min(100, max(result['fan_rated_power'] * 1.2, 60))
            result['fan_min_power'] = min(60, max(result['fan_min_power'] * 1.2, 30))
            result['rule'] = 'hot'
            return result
    mean = sum_temps / count
    if (mean < coolpoint) and (data['cp_status'] == 1):  # 如果机房偏冷,且空调处于打开状态
        result['fd_degree_set'] = 80  # result['fd_degree_set'] * 0.9
        result['fan_rated_power'] = max(result['fan_rated_power'] * 0.9, 60)
        result['fan_min_power'] = max(result['fan_min_power'] * 0.9, 30)
        result['rule'] = 'cold'
    return result


data_rule = pd.read_csv('./test/rule_test_data-20210225.csv')
data_rule.dropna(how='any', inplace=True)

data_rule['racks'] = data_rule['group_temp']
data_rule['air_conditioner_id'] = data_rule['acu_id']
data_rule = data_rule.to_dict('records')
# train_process(data=data_train)
rst_data = []
for it in data_rule:
    hotpoint = it["hotpoint"] if "hotpoint" in it else 28
    coolpoint = it["coolpoint"] if "coolpoint" in it else 24
    result = air_conditioner_adjust(it, hotpoint, coolpoint)
    it["rule_y"] = [result["fan_rated_power"], result["fan_min_power"], result["fd_degree_set"]]
    it["rule"] = result["rule"]
    rst_data.append(it)
rst_df = pd.DataFrame.from_records(rst_data)
rst_df.to_csv("./test/rule_test_rst-20210225.csv")
"""
air_list = json_obj["air_list"]
hotpoint = json_obj["hotpoint"] if "hotpoint" in json_obj else 28
coolpoint = json_obj["coolpoint"] if "coolpoint" in json_obj else 24
"""
