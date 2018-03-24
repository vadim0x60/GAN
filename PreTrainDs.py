# coding: utf-8

import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from Constant import Constants
from load_data import StyleData
from torch.autograd import Variable
from ModelDefine import DsModel
from ModelDefine import Embed
import sys

def train(epoches, batch_size, train_data):
    # data = train_data
    print "start trainning..........", len(train_data[0])
    train_data = indexData2variable(train_data)
    # has become the cuda data
    Loss = []
    acc = []
    for i in range(epoches):
        print i
        if i % 100 == 0:
            nowloss = getclfloss(train_data,ds,criterion)
            nowacc = getclfacc(train_data,ds)
            Loss.append(nowloss)
            acc.append(nowacc)
            print "%d\t\tacc:%.4f\tloss:%.4f" % (i, nowacc, nowloss)
        count = 0
        while count < max(len(train_data[0]),len(train_data[1])):
            temp_data1 = train_data[0][count:count+batch_size]
            temp_data2 = train_data[1][count:count+batch_size]
            count += batch_size
            loss = Variable(torch.Tensor([0])).cuda()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            
            for seq in temp_data1:
                emb_seq = embedding(seq).unsqueeze(0).unsqueeze(0)
                y_pred = ds(emb_seq)
                loss += criterion(y_pred,Variable(torch.cuda.LongTensor([0])))
            for seq in temp_data2:
                emb_seq = embedding(seq).unsqueeze(0).unsqueeze(0)
                y_pred = ds(emb_seq)
                loss += criterion(y_pred,Variable(torch.cuda.LongTensor([1])))
                
            if loss.data[0] != 0:
                loss.backward()
            optimizer1.step()
            optimizer2.step()
    print "trainning finished.........."
    return Loss, acc


def indexData2variable(data): 
    temp_data = [[],[]]
    for i in range(2):
        temp_data[i] = [Variable(torch.LongTensor(seq)).cuda() for seq in data[i]]
    return temp_data


def getclfacc(train_data,d_model):
    """
    Args:
    trian_data : data[0] is the source seqs data[1] is the target style
    d_model    : ds model or d model to classfify two class
    """
    d_model.train(False)
    if type(train_data[0][0]) != type(Variable(torch.Tensor([1])).cuda()):
        print "type don't fit"
        train_data = indexData2variable(train_data)
    acc = 0
    for i in range(2):
        for s in train_data[i]:
            emb = embedding(s).unsqueeze(0).unsqueeze(0)
            if d_model(emb).topk(1)[1].data.cpu().numpy()[0] == i:
                acc += 1
    d_model.train(True)
    return acc*1.0/(len(train_data[0]) + len(train_data[1]))


def getclfloss(train_data,d_model,criterion):
    """
    Args:
    train_data: data[0] is the source seqs data[1] is the target style
    d_model   : ds model or d model to classfify two class
    criterion : crossentropy
    """
    count  = 0
    d_model.train(False)
    if type(train_data[0][0]) != type(Variable(torch.Tensor([1])).cuda()):
        train_data = indexData2variable(train_data)
    loss = 0
    for i in range(2):
        for s in train_data[i]:
            count += 1
            if count % 1000 == 0:
                print count
            emb = embedding(s).unsqueeze(0).unsqueeze(0)
            loss += criterion(d_model(emb),Variable(torch.cuda.LongTensor([i]))).data
    d_model.train(True)
    return loss[0]/(len(train_data[0]) + len(train_data[1]))

# in this code you just need to use train function is OK and try to adjust some parameters
if __name__ == "__main__":
    """
    you shuld use this this script in this way
    python PretrainDs.py <styledatafilename> <traindatafilename> <buildNewModel?> <ModelName> epoches

    for instance:
        python ./traindata./style(don't add .npy) ./traindata/trainDataOfindex.npy yes ./Model/Ds.pkl epoches
    """
    
    booldic = dict(yes=True, y=True, Y=True, Yes=True, YES=True, no=False, N=False, n=False, NO=False, No=False)
    style = StyleData()
    style.load(sys.argv[1])
    train_data = np.load(sys.argv[2])
    # 
#     train_data = [train_data[0][:10000], train_data[1][:10000]]
    const = Constants(style.n_words)
    buildNewModel = booldic[sys.argv[3]]
    if buildNewModel:
        ds = DsModel(embedded_size=const.Embedding_size,
                    num_in_channels=1,
                    hidden_size=const.Hidden_size,
                    kind_filters=const.Ds_filters,
                    num_filters=const.Ds_num_filters).cuda()
        ds = ds.cuda() if const.use_cuda else ds
    else:
        ds = torch.load(sys.argv[4]) if const.use_cuda else ds

    embedding = Embed(embedding_size=const.Embedding_size, n_vocab=const.N_vocab)
    embedding = embedding.cuda() if const.use_cuda else embedding
    optimizer1 = optim.Adam(ds.parameters(), const.Lr)
    optimizer2 = optim.Adam(embedding.parameters(), const.Lr)
    criterion = nn.CrossEntropyLoss()
    train(int(sys.argv[5]),10,train_data)
    torch.save(ds,sys.argv[4])
