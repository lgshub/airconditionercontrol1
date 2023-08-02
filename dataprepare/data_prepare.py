# -*- coding:UTF-8 -*-
import pandas as pd
import numpy as np
import gc
import json
import logging

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s')

def get_selected_data(filename, conds, usecols, keycolums):
    df = pd.read_csv(filename)
    logging.info("---get_selected_data latn_id--")
    logging.info(df[:5])
    df = df[df['latn_id'].map(lambda x: x == conds['latn_id'])] if 'latn_id' in df.columns else df
    logging.info("---get_selected_data dc_id--")
    logging.info(df[:5])
    df = df[df['dc_id'].map(lambda x: x == conds['dc_id'])] if 'dc_id' in df.columns else df
    logging.info("---get_selected_data room_id--")
    logging.info(df[:5])
    df = df[df['room_id'].map(lambda x: x in (conds['roomIDs']))] if 'room_id' in df.columns else df
    df = df[df['location'].map(lambda x: x == conds['location'])] if 'location' in df.columns else df
    logging.info("---get_selected_data--")
    logging.info(df[:5])
    # get 1 h
    df['time'] = df['time'].apply(lambda x: str(x)[:10])
    df = df.drop_duplicates(subset=keycolums,
                            keep='first').reset_index(drop='True')
    df = df.sort_values(by='time')
    return df[usecols]


def deal_jigui_data(jg_file, selectconditions, usecols1, keycolums, kt_dict):
    jg_df = get_selected_data(jg_file, selectconditions, usecols1, keycolums)
    # 删除异常值 IDCNONE <=0 >80
    jg_df.drop(jg_df[jg_df['temp'] == 'IDCNONE'].index, inplace=True)
    jg_df['temp'] = jg_df['temp'].astype('float16')
    jg_df.drop(jg_df[(jg_df['temp'] <= 1) | (jg_df['temp'] >= 80)].index, inplace=True)
    jg_df.reset_index(drop=True, inplace=True)

    jg_df['rack_id'] = jg_df['rack_id'].apply(lambda x: str(x)[:9])
    jg_df = jg_df.groupby(['time', 'room_id', 'rack_id'], as_index=False)['temp'].max()
    #jg_df['acu_id'] = jg_df['rack_id'].map(kt_dict)
    jg_df['map_id'] = jg_df['room_id'].apply(str) + jg_df['rack_id'].apply(str)
    jg_df['acu_id'] = jg_df['map_id'].map(kt_dict)

    # 生成空调对应的机柜的平均温度df1
    jg_mean_df = jg_df.groupby(['time', 'acu_id'])['temp'].max().reset_index()  # 陕西使用了最大值
    g = jg_df.groupby(['time', 'acu_id'])['temp']
    d = [list(group) for name, group in g]
    jg_mean_df['group_temp'] = d
    logging.info("---机柜温度表--")
    logging.info(jg_mean_df[:5])

    return jg_mean_df


def deal_dldy_data(dldy_file, selectconditions, usecols1, keycolums, kt_dict):
    df_dldy = get_selected_data(dldy_file, selectconditions, usecols1, keycolums)
    # 删除异常值 IDCNONE
    df_dldy.drop(df_dldy[df_dldy['active_power'] == 'IDCNONE'].index, inplace=True)
    df_dldy['active_power'] = df_dldy['active_power'].astype('float16')
    # 陕西1号机楼功率为0
    df_dldy.drop(df_dldy[df_dldy['ele_currnt'] == 'IDCNONE'].index, inplace=True)
    df_dldy['ele_currnt'] = df_dldy['ele_currnt'].astype('float16')
    df_dldy.drop(df_dldy[df_dldy['voltage'] == 'IDCNONE'].index, inplace=True)
    df_dldy['voltage'] = df_dldy['voltage'].astype('float16')
    df_dldy['active_power'] = df_dldy['ele_currnt'] * df_dldy['voltage']

    df_dldy.reset_index(drop=True, inplace=True)

    # 部分电流出现负值波动，功率统一归为 0
    df_dldy['active_power'] = np.where(df_dldy['active_power'] < 0, 0, df_dldy['active_power'])
    # 2021042817之前之间的功率单位是W,要转换为KW
    df_dldy['active_power'] = np.where(df_dldy['active_power'] > 10, df_dldy['active_power'] / 1000,
                                       df_dldy['active_power'])

    # 分别将 空调1，机柜3拿出来
    df_dev_kt = df_dldy[df_dldy['dev_type'] == 1][['time', 'dev_id', 'active_power']]
    df_dev_kt.columns = ['time', 'acu_id', 'acu_power']
    df_dev_kt['acu_id'] = df_dev_kt['acu_id'].apply(str)
    # df_dev_1.to_csv('acu_power.csv',index=False)

    df_dev_jg = df_dldy[df_dldy['dev_type'] == 3]
    df_dev_jg['dev_id'] = df_dev_jg['dev_id'].apply(lambda x: str(x)[:9])
    #df_dev_jg['acu_id'] = df_dev_jg['dev_id'].map(kt_dict)
    df_dev_jg['map_id'] = df_dev_jg['room_id'].apply(str) + df_dev_jg['dev_id'].apply(str)
    df_dev_jg['acu_id'] = df_dev_jg['map_id'].map(kt_dict)

    df_dev_jg = df_dev_jg.groupby(['time', 'acu_id'])['active_power'].sum().reset_index()
    logging.info("---电流电压表处理--")
    logging.info(df_dev_jg[:5])
    logging.info(df_dev_kt[:5])

    return df_dev_jg, df_dev_kt


def deal_kt_data(kt_file, selectconditions, usecols1, keycolums):
    df_kt1 = get_selected_data(kt_file, selectconditions, usecols1, keycolums)
    df_kt1['acu_id'] = df_kt1['acu_id'].apply(lambda x: str(x))#[:7]
    # 删除异常值 IDCNONE
    df_kt1.drop(df_kt1[df_kt1['return_t'] == 'IDCNONE'].index, inplace=True)
    df_kt1.reset_index(drop=True, inplace=True)
    logging.info("---空调表处理--")
    logging.info(df_kt1[:5])
    return df_kt1


def merge_tables(path, kt_dict, selectconditions):
    jg_file = path + '7.机柜温度数据.csv'
    usecols1 = ['time', 'room_id', 'rack_id', 'location', 'test_height', 'temp']
    keycolums = ['time', 'room_id', 'rack_id', 'location', 'test_height']
    jg_mean_df = deal_jigui_data(jg_file, selectconditions, usecols1, keycolums, kt_dict)

    dldy_file = path + '5.电流电压数据.csv'
    usecols1 = ['time', 'room_id', 'dev_type', 'dev_id', 'ele_currnt', 'voltage', 'active_power']
    keycolums = ['time', 'room_id', 'dev_type', 'dev_id']
    df_dev_jg, df_dev_kt = deal_dldy_data(dldy_file, selectconditions, usecols1, keycolums, kt_dict)

    kt_file = path + "4.空调数据.csv"
    usecols1 = ['time', 'room_id', 'acu_id', 'setting_t', 'setting_h',
                'cp_status', 'fan_speed', 'return_t', 'fd_status',
                'fd_degree', 'mode']
    keycolums = ['time', 'room_id', 'acu_id']
    df_kf = deal_kt_data(kt_file, selectconditions, usecols1, keycolums)

    qx_file = path + "13.气象数据.csv"
    usecols1 = ['time', 'outdoor_t', 'outdoor_h']
    keycolums = ['time']
    df_qx = get_selected_data(qx_file, selectconditions, usecols1, keycolums)
    ## 合并
    logging.info(jg_mean_df['time'].drop_duplicates())
    logging.info(df_kf['time'].drop_duplicates())
    jg_mean_df.to_csv(path + "/data202106_7s_jg_mean_df.csv")
    df_kf.to_csv(path + "/data202106_7s_df_kf.csv")
    df_all = df_kf.merge(jg_mean_df, on=['time','acu_id'], how='left')  # 机柜温度
    logging.info("---合并后表为0-----")
    logging.info(type(df_kf['acu_id'][0]))
    logging.info(df_all[:5])
    df_all = df_all.merge(df_dev_jg, on=['time', 'acu_id'], how='left')  # 机柜功率
    df_all = df_all.merge(df_dev_kt, on=['time', 'acu_id'], how='left')  # 空调自己的功率
    df_all = df_all.merge(df_qx, on=['time'], how='left')

    ## 删除IDCNONE
    df_all = df_all.replace('IDCNONE', np.nan)
    df_all = df_all.dropna(axis=1, how='all')  # 删除全为none的列
    logging.info("---合并后表为1--")
    logging.info(df_all[:5])
    df_all.dropna(how='any', inplace=True)  # 删除存在none的行
    df_all.reset_index(drop=True, inplace=True)
    logging.info("---合并后表为2--")
    logging.info(df_all[:5])

    return df_all


def main():
    latn_id =  86101#83502
    dc_id =  'SN-YJD-XX'#936416
    #roomIDs = ['HXXID1', 'HXXID2', 'HXXID3', 'HXXID4', 'HXXID5']#[9129, 10129, 11129, 13129, 15129, 17129, 20129]#[2, 3, 4, 14, 15]
    roomIDs = [2, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    location = 0  # 0:front, 1:back
    rawdata_path = '../data/raw_data/data-20211121-20211221-861/' #'../data/raw_data/data-20210615-861/'
    dic_path = './room_dic/shanxi/shanxi_15s_dict.txt' #'./room_dic/xiamen/xiamen_5dict.txt'
    conditions = {"latn_id": latn_id, "dc_id": dc_id, "roomIDs": roomIDs, "location": location}
    kt_dict = json.loads(open(dic_path, 'r', encoding='utf-8').read())
    df_all = merge_tables(rawdata_path, kt_dict, conditions)
    df_all.to_csv(rawdata_path + "/data20211223_15s_raw.csv")



if __name__ == '__main__':
    main()
