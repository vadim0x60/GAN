# coding: utf-8
from io import open
import string
import glob
import re
import random

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class StyleData(object):
    # noinspection SpellCheckingInspection
    def __init__(self, data=[]):
        self.word2index = {}
        self.index2word = {0: 'Eos', 1: 'Sos'}
        self.n_words = 2
        self.word2count = {}
        self.target_style = 1
        for stype in data:
            for seq in stype:
                self.addSequence(seq)

    def addSequence(self, seq):
        for word in seq:
            self.addWord(word)

    def addWord(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2count[word] = 1
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def save(self, name):
        narray = np.array([self.word2index, self.index2word, self.n_words, self.word2count, self.target_style])
        np.save(name, narray)

    def load(self, name):
        narray = np.load(name + '.npy')  # attention
        self.word2index = narray[0]
        self.index2word = narray[1]
        self.n_words = narray[2]
        self.word2count = narray[3]
        self.target_style = narray[4]


def data2index(data, style):
    # we just need to pass the data in such format
    # the first dim is about the category 0 represent source 1 represent the target
    # the second dim is the seqs belong to the certain style
    # so there are just two dim and the seqs shuld be a list of words
    data = [
        [seq2index(seq, style, i) for seq in data[i]] for i in range(len(data))
    ]
    return data


def seq2index(seq, style, index):
    data = [style.word2index[word] for word in seq]
    data.append(1)
    return data


# example to use this code
# style = StyleData(data)
# data = data2index(data)
# if you want to reuse the data later you shuld use code:
# style.save(<filename>)
# np.save(<filename>,data)
# if you want to load them again
# style = StyleData()
# style.load(<filename>) // filename don't need to add .npy
# data = np.load(<filename>) // filename need to add .npy

if __name__ == "__main__":
    print('finished')
