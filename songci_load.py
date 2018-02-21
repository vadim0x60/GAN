# coding: utf-8
import numpy as np
import random
import re
import langconv as lng
import zh_wiki as zh
import sys
import jieba
import sqlite3 as sql
import load_data as ld


def readsongci(filename):
    db = sql.connect(filename)
    data = db.cursor()
    ci = data.execute("select * from " + re.split(r'[\/\.]',filename)[-2])
    songci = []
    for seq in ci:
        songci.append(seq[3].strip().split('\n'))
    return songci


def ciandpoet2seq(poets):
    data = []
    for p in poets:
        for seq in p:
            data.append([c for c in seq])
    return data    


if __name__ == "__main__":
    filename = "./data/ci.db" if len(sys.argv) < 2 else sys.argv[1]
    data = readsongci(filename)
    seqs = ciandpoet2seq(data)
    datas = np.load('./data/wikiseqdata.npy')[:len(seqs)]
    style = ld.StyleData([datas,seqs])
    seqdata = ld.data2index([datas,seqs],style=style)
    style.save('./data/wiki2songci')
    array = np.array(seqdata)
    np.save('./data/traindatawiki2songci',array)
