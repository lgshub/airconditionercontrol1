from heatmap import *


class JFHeatMap(HeatMap):
    def get_label(self, data_map, name, rack_id, min_val, max_val):
        for i in range(data_map.shape[0]):
            for j in range(data_map.shape[1]):
                if data_map[i, j] is not None:
                    if data_map[i, j] == -1:
                        name[i, j] = "IDCNONE"
                    elif data_map[i, j] < 0:
                        name[i, j] = "负值异常"
                    elif data_map[i, j] == 0:
                        name[i, j] = "零值"
                    elif data_map[i, j] < min_val:
                        name[i, j] = "低温异常" + '(%.2f)' % data_map[i, j]
                    elif data_map[i, j] < max_val:
                        name[i, j] = '(%.2f)' % data_map[i, j]
                    else:
                        name[i, j] = '(%.2f)' % data_map[i, j]
        return name


def checkpathexist(file):
    if not os.path.exists(file):
        os.makedirs(file)


def main():
    notchecked = False
    datapath = './data4heatmap/'
    JFname = '陕西1号楼'
    daystamp = '20210615-861'
    roomtopology = "1号楼平面图(替换编号后).xlsx"
    location = 0  # cold 0  hot 1
    roomiddic = {"3-1": "9129", "3-2": "10129", "3-3": "11129", "3-5": "13129", "4-1": "15129",
                 "4-3": "17129", "4-4": "18129", "4-6": "20129"}
    draw_topology_img = True
    time = '2021061322'  # 绘制图的时间，check 中间文件后填写
    temp_min_val, temp_max_val = 12, 36
    in_data_path = datapath + JFname + './input/data-' + daystamp + '/'
    in_loc_file = datapath + JFname + './input/' + roomtopology
    out_heat_doc_file = datapath + JFname + './output/' + daystamp + '/' + JFname + '_' + daystamp + '.docx'
    out_heat_jg = datapath + JFname + './output/' + daystamp + '/jp_heatmaps/'
    out_mid_data = datapath + JFname + './output/' + daystamp + '/mid_data/'
    checkpathexist(out_heat_jg)
    checkpathexist(out_mid_data)

    JF = JFHeatMap(in_data_path, in_loc_file, out_heat_jg, out_heat_doc_file,
                  out_mid_data, roomiddic)
    if notchecked:
        JF.process_loc()
        print(JF.get_room_id())
        JF.process_data(location)
    else:
        room_ID = JF.get_room_id()
        room_ID.remove("18129")
        JF.plot_headmap(time, room_ID, temp_min_val=temp_min_val, temp_max_val=temp_max_val)
        if JF.room_id_dic:
            room_name = dict(map(reversed, JF.room_id_dic.items()))
        else:
            room_name = None
        if draw_topology_img:
            JF.generate_docs_with_topology(time, room_ID, room_name=room_name)
        else:
            JF.generate_docs(time, room_ID, room_name=room_name)


if __name__ == '__main__':
    main()
