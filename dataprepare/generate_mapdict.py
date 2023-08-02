import pandas as pd
import numpy as np
import json

def gendict(rstdic, roomid, keys, colnums, numup, numdown, spl):
    bjdic = {}
    start = 101+numup
    for k, ns in zip(keys[:spl], colnums[:spl]):
        bjdic[k] = []
        for ni in range(ns):
            bjdic[k] += [i for i in range(start, start+numdown)]
            start = start + 100

    start = 101
    for k, ns in zip(keys[spl:], colnums[spl:]):
        bjdic[k] = []
        for ni in range(ns):
            bjdic[k] += [i for i in range(start, start+numup)]
            start = start + 100
    #correct = {}
    for k in bjdic:
        for v in bjdic[k]:
            rstdic[str(roomid)+str(v)] = k


def main():
    rstdic = {}
    #3-1
    keys = [str(i)+"0081129" for i in range(1,9)]
    colnums = [1, 2, 2, 3, 3, 2, 2, 1]
    numup, numdown, spl, roomid = 9, 10, 4, 9129
    gendict(rstdic, roomid, keys, colnums, numup, numdown, spl)
    #3-2
    keys = [str(i) + "0091129" for i in range(1, 10)]
    colnums = [1, 2, 2, 2, 1, 1, 2, 2, 3]
    numup, numdown, spl, roomid = 8, 10, 5, 10129
    gendict(rstdic, roomid, keys, colnums, numup, numdown, spl)
    #3-3 0020099129
    keys = [str(i) + "0099129" for i in range(2, 5)] + ["10099129"] + [str(i) + "0099129" for i in range(5, 9)]
    colnums = [2, 2, 1, 3, 1, 2, 2, 2]
    numup, numdown, spl, roomid = 9, 10, 4, 11129
    gendict(rstdic, roomid, keys, colnums, numup, numdown, spl)
    #3-5 0020099129
    keys = [str(i) + "0118129" for i in range(6, 11)] + ["20118129", "10118129"] + [str(i) + "0118129" for i in range(3, 6)]
    colnums = [1, 1, 1, 2, 3, 2, 2, 2, 1, 1]
    numup, numdown, spl, roomid = 6, 10, 5, 13129
    gendict(rstdic, roomid, keys, colnums, numup, numdown, spl)
    #4-1 0010133129
    keys = [str(i) + "0133129" for i in range(1, 9)]
    colnums = [2, 1, 2, 3, 3, 2, 2, 1]
    numup, numdown, spl, roomid = 9, 10, 4, 15129
    gendict(rstdic, roomid, keys, colnums, numup, numdown, spl)
    #4-3 0010150129
    keys = [str(i) + "0150129" for i in range(1, 9)]
    colnums = [3, 1, 1, 2, 1, 2, 2, 3]
    numup, numdown, spl, roomid = 9, 10, 4, 17129
    gendict(rstdic, roomid, keys, colnums, numup, numdown, spl)

    #4-6 0050173129
    keys = [str(i) + "0173129" for i in range(5, 9)] + [ str(i) + "0173129" for i in range(1, 5)]
    colnums = [2, 2, 2, 2, 1, 2, 2, 3]
    numup, numdown, spl, roomid = 9, 10, 4, 20129
    gendict(rstdic, roomid, keys, colnums, numup, numdown, spl)
    """
    rstdict = {}
    for k in bjdic:
        for v in bjdic[k]:
            rstdict[v] = k
    """

    json.dump(rstdic, open("room_dic/shanxi/dict_num1.txt", "w"))



if __name__ == '__main__':
    main()

