from heatmap import *


class JFHeatMap(HeatMap):
    pass



def checkpathexist(file):
    if not os.path.exists(file):
        os.makedirs(file)


def main():
    notchecked = False
    datapath = './data4heatmap/'
    JFname = '陕西'
    daystamp = '20210526-861'
    roomtopology = "loc_all.xlsx"
    location = 0  # cold 0  hot 1
    roomiddic = None
    draw_topology_img = True
    time = '2021052420' # 绘制图的时间，check 中间文件后填写
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
