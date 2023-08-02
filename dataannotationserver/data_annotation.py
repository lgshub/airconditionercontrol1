import pandas as pd
import random
import logging

logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s')


def rule_hot(cp_status, fan_speed, temp, activePower):
    fan_rated_power = ''
    leveldic = {0: [0, 0, 0], 1: [60, 40, 80], 2: [80, 50, 80], 3: [100, 60, 80]}
    level = 1
    if cp_status == 0:
        if activePower >= 20:
            level = 1
        else:
            level = 0
    else:  # cp_status = 1
        if activePower >= 42:
            if temp >= 32:
                level = 3
            else:
                if fan_speed > 80:
                    level = 3
                else:
                    fan_rated_power = 80
                    level = 2
        elif activePower > 35:
            if temp >= 35:
                level = 3
            elif temp >= 31:
                if fan_speed > 80:
                    level = 3
                else:
                    level = 2
            else:
                if fan_speed >= 65:
                    level = 2
                else:
                    level = 1
        elif activePower > 15:
            if temp >= 29:
                if fan_speed > 60:
                    level = 2
                else:
                    level = 1
            else:
                level = 1
        else:
            if temp >= 35:
                level = 2
            elif temp >= 30:
                level = 1
            else:
                level = 0
    return leveldic[level]


def rule_cold(cp_status, fan_speed, temp, activePower):
    fan_rated_power = ''
    leveldic = {0: [0, 0, 0], 1: [60, 30, 80], 2: [80, 40, 80], 3: [100, 50, 80]}
    level = 1
    if cp_status == 0:
        if activePower >= 20:
            level = 1
        else:
            level = 0
    else:  # cp_status = 1
        if activePower >= 40:
            if temp >= 26:
                level = 3
            else:
                if fan_speed >= 80:
                    level = 3
                else:
                    fan_rated_power = 80
                    level = 2
        elif activePower > 30:
            if temp >= 25:
                if fan_speed >= 80:
                    level = 3
                else:
                    level = 2
            else:
                if fan_speed >= 70:
                    level = 2
                else:
                    level = 1
        elif activePower > 1:
            if temp >= 24:
                if fan_speed >= 65:
                    level = 2
                else:
                    level = 1
            else:
                level = 1
        else:
            level = 0
    return leveldic[level]


def annotation(data, outtrainfile, outtestfile, outannotationfile):
    # data = pd.read_csv(infile)#, encoding="gbk"
    data.dropna(how='any', inplace=True)  # 删除存在none的行
    data_x = data[['cp_status', 'fan_speed', 'temp', 'active_power']]
    fan_rated_power_list, fan_min_power_list, fd_degree_set_list = [], [], []
    for index, row in data_x.iterrows():
        fan_rated_power, fan_min_power, fd_degree_set = rule_cold(row['cp_status'], row['fan_speed'], row['temp'],
                                                                  row['active_power'])
        fan_rated_power_list.append(fan_rated_power)
        fan_min_power_list.append(fan_min_power)
        fd_degree_set_list.append(fd_degree_set)
    data['fan_rated_power'], data['fan_min_power'], data[
        'fd_degree_set'] = fan_rated_power_list, fan_min_power_list, fd_degree_set_list
    data = data.sample(frac=1)
    dflen = len(data)
    logging.info(dflen)
    data.iloc[0:int(dflen * 0.9), :].to_csv(outtrainfile)
    logging.info(len(data.iloc[0:int(dflen * 0.9), :]))
    data.iloc[int(dflen * 0.9):, :].to_csv(outtestfile)
    data.to_csv(outannotationfile)
    logging.info(len(data.iloc[int(dflen * 0.9):, :]))
