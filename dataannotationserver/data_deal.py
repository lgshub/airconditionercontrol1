import os
import sys
import pandas as pd
import logging
from annotation import *

sys.path.append('../')
from config.constants import CONVERTERS

logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s')


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield root + '/' + f


def main():
    inpath = '../data/raw_data/data-20211121-20211221-861/data20211223_15s_raw.csv' #'../data/raw_data/data-20211101-20211114-835-936416/data20211101_5s_raw.csv'
    outannotationfile = '../data/annotation_data/shanxi/data20211223_15s_anno.csv'
    outtrainfile = '../data/train_data/shanxi/data20211223_15s_train.csv'
    outtestfile = '../data/test_data/shanxi/data20211223_15s_test.csv'

    dfall = pd.read_csv(inpath, converters=CONVERTERS)
    logging.info("--输入数据--")
    logging.info(dfall[:5])
    annotation(dfall, outtrainfile, outtestfile, outannotationfile)


if __name__ == '__main__':
    main()
