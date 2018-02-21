# coding: utf-8

import numpy as np
import re
import string
import json
from io import open
import langconv
import zh_wiki
import glob
import sys
import load_data as ld


def read_poet(filename):
    data = []
    poetsjson = open(filename)
    poets = json.load(poetsjson)               
    for p in poets:
        data.extend(p['paragraphs'])
    return data


def poet2seq(poets):
    data = []
    data = [[c for c in seq] for seq in poets]
    return data


if __name__ == "__main__":
    filenames = glob.glob('./data/poet/poet.tang.*.json') # I don't add song poem in
    data = []
    for fname in filenames:
        data.extend(poet2seq(read_poet(fname)))
    datas = np.load('./data/wikiseqdata.npy')
    style = ld.StyleData([datas,data])
    indexdata = ld.data2index([datas,data],style)
    array = np.array(indexdata)
    np.save('./data/traindatawiki2poet',array)
    style.save('./data/wiki2poerstyle')

