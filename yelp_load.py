import numpy as np
from io import open
import re
import string
import random
from Constant import Constants
from load_data import data2index, StyleData


def readfile(filename):
    con = Constants(100)
    data = open(filename, encoding='utf-8').read().strip().split('\n')
    data = [split(seq.lower()) for seq in data]
    # do we need to constraint the length or just padding is enough
    data = [s for s in data if con.Max_len > len(s) > con.Min_len]
    return data


def split(seq):
    return re.split(r' ', seq.strip())


def saveTrainData(filenames):
    # src represent source and trt represent target
    srcname = filenames[0]
    trtname = filenames[1]
    srcdata = readfile(srcname)
    trtdata = readfile(trtname)
    datapairs = [srcdata, trtdata]
    style = StyleData(datapairs)
    style.save('./traindata/style')
    datapairs = data2index(datapairs, style)
    datapairs = np.array(datapairs)
    np.save('./traindata/trainDataOfIndex', datapairs)
    return


if __name__ == "__main__":
    filenames = ['./data/yelp/sentiment.train.0', './data/yelp/sentiment.train.1']
    saveTrainData(filenames)
    print("finished")
