#训练数据1203类别数3array(['60,40,100', '60,30,100', '100,50,60'], dtype=object)

训练服务：
输入字段：['outdoor_t', 'outdoor_h', 'temp', 'activePower', 'setting_t', 'setting_h',
                       'cp_status', 'return_t', 'label', 'fan_rated_power', 'fan_min_power', 
                  'fd_degree_set']
例：
time               2020120312
room_id                HXXID3
acu_id             艾特网能空调1725
setting_t                  25
setting_h                  50
cp_status                   1
fan_speed                  58
return_t                 26.8
fd_status                   1
fd_degree                  59
mode                        1
temp                  24.5389
activePower           46997.6
acu_power             2421.78
outdoor_t                26.6
outdoor_h                44.5
open                        1
fan_rated_power            60
fan_min_power              40
fd_degree_set             100

输出：模型文件'lgb_model.pkl'
#########################################################
预测服务：
输入字段：['time', 'room_id', 'acu_id', 'setting_t', 'setting_h', 'cp_status',
       'fan_speed', 'return_t', 'fd_status', 'fd_degree', 'mode', 'temp',
       'activePower', 'acu_power', 'outdoor_t', 'outdoor_h']

输出label：array(['60,30,100', '100,50,60', '60,30,100'], dtype=object)
#含义为'fan_rated_power', 'fan_min_power', 
                  'fd_degree_set'