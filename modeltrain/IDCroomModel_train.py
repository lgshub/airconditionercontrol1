import numpy as np
import pandas as pd
import random
from lightgbm import LGBMClassifier, plot_importance
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, accuracy_score
# from pylab import *
import joblib
import logging
import os

# python IDCroomModel_train.py

logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s')


# data_train.columns is  ['setting_t', 'setting_h', 'cp_status','fan_speed',
# 'return_t', 'fd_degree', 'outdoor_t','outdoor_h', 'temp', 'activePower','kt_power',
# 'fan_rated_power', 'fan_min_power', 'fd_degree_set']

# idxs = ['rack_h', 'rack_l', 'rack_avg', 'env_avg', 'env_h', 'temp', 'return_t']
idxs = ['rack_h', 'rack_l', 'rack_avg', 'env_avg', 'env_h', 'return_t']
# idxs = ['rack_avg', 'env_avg', 'env_h', 'return_t']
# idxs = ['env_avg', 'env_h', 'return_t']
# idxs = []


def _label(data, choice=-1):
    idx = data.columns
    conditioner = {'setting_t': [], 'return_t': [],
                   'power': [], 'rack_avg': []}
    for a in idx:
        if '_s' in a:
            conditioner['setting_t'].append(data[a])
        elif '_r' in a:
            conditioner['return_t'].append(data[a])
        # elif '_p' in a:
        #     conditioner['power'].append(data[a])
        # elif '_h' in a:
        #     conditioner['rack_area'].append(data[a])
    # assert len(conditioner['setting_t']) == 1
    data['temp'] = data['env_avg']
    data['setting_t'] = conditioner['setting_t'][choice]
    data['return_t'] = conditioner['return_t'][choice]
    # data['rack_avg'] = conditioner['rack_avg'][choice]
    return data[idxs + ['low', 'high', '控制范围', '环境最高温', '机柜最高温', '机柜预警线', '机柜报警线', '回风预警线',
                        '环境报警线'] + ['setting_t']]


def data_process(data_train):
    data_train['label'] = data_train['setting_t'].apply(str)

    # 看起来回风温度最重要？
    # train_data = data_train[['return_t', 'label']]
    # train_data = data_train[['return_t', 'temp', 'label']]
    # train_data = data_train[['return_t', 'rack_avg', 'label']]
    train_data = data_train[idxs + ['low', 'high', '环境最高温', '机柜最高温', '机柜预警线', '机柜报警线', '回风预警线',
                                    '环境报警线'] + ['label']]
    # train_data = data_train[idxs + ['label']]
    # train_data = data_train[idxs + ['机柜预警线' , '机柜报警线', '回风预警线', \
    #     '环境报警线'] + ['label']]
    # train_data = data_train[idxs + ['机柜预警线' , '机柜报警线', '回风预警线', \
    #     '环境报警线'] + ['label']]

    for col_name in train_data.columns[:-1]:
        train_data.loc[:, col_name] = train_data.loc[:, col_name].fillna(
            train_data.loc[:, col_name].mean())  # 均值填充

    train_data.dropna(inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(train_data.iloc[:, :-1],
                                                        train_data['label'],
                                                        test_size=0.1,
                                                        random_state=1)

    return x_train, x_test, y_train, y_test


def train_process(data_train, model_output_file):
    x_train1, x_test1, y_train1, y_test1 = data_process(data_train)
    # parameters = {
    #     'learning_rate': [0.005, 0.01, 0.05, 0.1],
    #     'n_estimators': range(400, 900, 100),
    # }

    # gbm = LGBMClassifier(objective='multiclass',
    #                      max_depth=6,
    #                      num_leaves=40,
    #                      learning_rate=0.01,
    #                      min_child_samples=20,
    #                      min_child_weight=0.001,
    #                      reg_alpha=0.001,
    #                      reg_lambda=8,
    #                      n_estimators=400
    #                      )
    # gsearch = GridSearchCV(gbm, param_grid=parameters,
    #                        scoring='f1_weighted', cv=3)
    # gsearch.fit(x_train1, y_train1)
    # logging.info('参数的最佳取值:{0}'.format(gsearch.best_params_))
    # logging.info('最佳模型得分:{0}'.format(gsearch.best_score_))
    # logging.info(gsearch.cv_results_['mean_test_score'])
    # logging.info(gsearch.cv_results_['params'])
    # learning_rate_cv = gsearch.best_params_['learning_rate']
    # n_estimators_cv = gsearch.best_params_['n_estimators']

    learning_rate_cv = 0.1
    n_estimators_cv = 200

    lgb = LGBMClassifier(objective='multiclass',
                         max_depth=-1,
                         num_leaves=31,
                         learning_rate=learning_rate_cv,
                         min_child_samples=20,
                         min_child_weight=0.0,
                         reg_alpha=0.5,
                         reg_lambda=0.5,
                         n_estimators=n_estimators_cv,
                         )

    kf = KFold(2, shuffle=True, random_state=np.random.randint(1000))

    for k, (train_index, val_index) in enumerate(kf.split(x_train1, y_train1)):
        x_train = x_train1.iloc[train_index]
        y_train = y_train1.iloc[train_index]
        x_val = x_train1.iloc[val_index]
        y_val = y_train1.iloc[val_index]
        lgb.fit(x_train, y_train, eval_set=[
            (x_val, y_val)], early_stopping_rounds=0, verbose=0)
    results = cross_val_score(lgb, x_train1, y_train1, cv=kf)
    logging.info("Train Accuracy: %.2f%% (%.2f%%)" %
                 (results.mean() * 100, results.std() * 100))
    y_pred = lgb.predict(x_test1)
    #
    accuracy = accuracy_score(y_test1, y_pred)
    logging.info("Test Accuracy: %.2f%%" % (accuracy * 100.0))
    joblib.dump(lgb, model_output_file)


model_file = '../model/AC_Model_HB2.pkl'


if 0:
    data = pd.read_csv('base0.csv')
    data = data.loc[np.random.randint(0, data.__len__()-1, 30000)]
    train_process(data, model_file)

elif 1:
    data = pd.read_csv('base0.csv')
    # 调整数据点比例
    picked_idx = []
    for T in range(21, 30):
        idx = data[data['setting_t'] == T].index
        picked_idx += np.random.choice(idx, 3500).tolist()

    # print(picked_idx)
    
    data = data.loc[picked_idx]
    data.index = range(1, len(data) + 1)
    
    data.to_csv('base.csv')
    train_process(data, model_file)

else:
    # make base0
    ch = 0
    # data_train = pd.read_csv('../data/train_data/待打标数据汉口新区金银湖IDC_12机房202206010614.csv')
    # file = ['绍_C_绍兴齐贤局_2F综合机房.csv','婺_A_云计算中心二号楼局_3F3-2电力电池室.csv', \
    #     '甬_C_余姚富巷母局_2F综合机房.csv', '禾_C_海宁干河街局_4F政府云机房.csv', \
    #         '禾_D_海盐武原城南局_2F综合机房.csv']
    data = None
    record = pd.read_csv('浙江强化记录.csv')
    record.dropna(axis=0, how='all', inplace=True)

    # 选出空调数量为1个的机房
    # record = record.loc[record[record['空调数量'] == '1'].index]
    # record.index = range(len(record))

    # 将控制范围分成low和high
    ctrl = record['控制范围']
    ctrl1 = ctrl.copy(deep=1)
    for i in range(ctrl1.__len__()):
        ctrl1[i] = ctrl1[i][:2]
    ctrl2 = ctrl.copy(deep=1)
    for i in range(ctrl2.__len__()):
        ctrl2[i] = ctrl2[i][-2:]
    record['high'] = ctrl2
    record['low'] = ctrl1

    lines = ['low', 'high', '控制范围', '机柜最高温',
             '环境最高温', '机柜预警线', '机柜报警线', '回风预警线', '环境报警线']
    rec = record[['low', 'high', '控制范围', '数据中心', '机房名称', '机柜最高温', '环境最高温', '机柜预警线', '机柜报警线',
                  '回风预警线', '环境报警线']].dropna(axis=0, inplace=False)

    # 节能率大于0的机房
    saved = pd.read_csv('save.csv').to_numpy()

    file_path_ = os.listdir('../data2/202306120947/')

    # 机房dict
    names = {}
    for i in rec.index:
        names[rec.loc[i, '数据中心'] + '_' + rec.loc[i, '机房名称']] = i

    # 是否强化记录里的也被节能
    _use = [c if (c in saved and c+'.csv' in file_path_)
            else None for c in names.keys()]

    for file_ in _use:
        if file_ == None:
            continue
    # for i in train_idx:
        file_ = file_ + '.csv'
        path = '../data/data2/202306120947/' + file_
        data_train = pd.read_csv(path)
        name = file_[:-4]

        data_train.dropna(axis=1, how='all', inplace=True)
        data_train.dropna(axis=0, how='any', inplace=True)

        # 在后面补上强化记录里的指标
        if name in names.keys():
            idx = names[name]
            for v in lines:
                # 对一列赋值
                data_train[v] = rec.loc[idx][v]
                # pd.merge(data_train, rec.loc[idx][2:])
        else:
            assert 1 == 0

        if data is not None:
            data = pd.concat([data, _label(data_train, ch)])
        else:
            data = _label(data_train, ch)

    data.index = range(1, len(data) + 1)

    data = data.drop(data[(data['setting_t'] < 21) |
                     (data['setting_t'] > 29)].index)
    
    # 调整数据点比例
    # picked_idx = []
    # for T in range(21, 30):
    #     idx = data[data['setting_t'] == T].index
    #     picked_idx += np.random.choice(idx, 3500).tolist()

    # print(picked_idx)
    
    # data = data.loc[picked_idx]
    # data.index = range(1, len(data) + 1)
    
    data.to_csv('base0.csv')
    # train_process(data, model_file)
