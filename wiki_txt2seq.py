# coding: utf-8
import langconv as lng
import zh_wiki
import re
import string
from io import open
import jieba
import numpy as np
from glob import glob


def readfile(filename):
    convert = lng.Converter('zh-hans').convert # this is a function
    filetxt = open(filename,encoding='utf-8').read().strip()
    filetxt = re.split(u'[。\n]',filetxt)
    filetxt = [convert(s)+u'。' for s in filetxt if len(s) > 7]
    return filetxt


def text2seq(data):
    data = [list(jieba.cut(s)) for s in data if len(list(jieba.cut(s))) > 7]
    return data


def savedata(data):
    data = np.array(data)
    np.save('./data/wikiseqdata',data)


if __name__ == "__main__":
    filenames = glob('./data/std_wiki*')
    savedata(text2seq(readfile(filenames[0])))


