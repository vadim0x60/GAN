# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random 
from Constant import Constants
from load_data import StyleData
from PreTrainDs import indexData2variable

gan = torch.load('gan.pkl')


style = StyleData()
style.load('all_style')
const = Constants(style.n_words)
train_data = np.load('trainDataOfIndex.npy')
optimizer = optim.Adam(gan.parameters(),lr=const.Lr)
ds = torch.load('Ds.pkl')
ds_emb = torch.load('embedding.pkl')
lamda1 = 1
lamda2 = 1
lamda3 = 3
cross_entropy = nn.CrossEntropyLoss()

def trainVAE_D(epoches,batch_size,data,ds_model,ds_emb):
    ds_model.train(False)
    ds_emb.train(False)
    train_data = indexData2variable(data)
    for i in range(epoches):
        shuffleData(train_data)
        train_data = build2pairs(train_data)
#         if i % 100 == 0:
#             print getTypeLoss()
        count = 0
        while count < len(train_data):
            tempdata = train_data[count:count+batch_size]
            
            if tempdata == []:
                break
                
            count += batch_size
            optimizer.zero_grad()
            Lrec = 0
            Lcyc = 0
            Ldis = 0
            Ladv = 0
            optimizer.zero_grad()
            Loss = 0
            if random.choice([0]):
                for seqs in tempdata:
                    dic = gan(seqs[0],seqs[1],D_train=False)
#                     print dic
                    Lrec = cross_entropy(dic['x1_hat_noT'],seqs[0])+cross_entropy(dic['x2_hat_noT'],seqs[1])
                    Lcyc = cross_entropy(dic['x1_bar_noT'],seqs[0])+cross_entropy(dic['x2_bar_noT'],seqs[1])
                    emb = ds_emb(seqs[0]).unsqueeze(0).unsqueeze(0)
                    Ldis = (ds_model(emb)[0][1]*(dic['y1']-dic['y_star'])**2).sum()
#                     Ladv = -(torch.log(1-dic['D_x1_wl'][0][1]) + torch.log(dic['D_x2_hat'][0][1]))
                    Ladv = -cross_entropy(dic['D_x1_wl'],Variable(torch.LongTensor([0])))
                    Loss += Lrec + lamda2*Lcyc + lamda3*Ldis - lamda1*Ladv
            else:
                for seqs in tempdata:
                    dic = gan(seqs[0],seqs[1],Ez_train=False,Ey_train=False,G_train=False,
                              Lcyc=False, Lrec=False, Ldis = False)
#                     Ladv = -(torch.log(1-dic['D_x1_wl'][0][1]) + torch.log(dic['D_x2_hat'][0][1]))
                    Ladv = cross_entropy(dic['D_x1_wl'],Variable(torch.LongTensor([0])))+ cross_entropy(dic['D_x2_hat'],Variable(torch.LongTensor([1])))
                    Loss += lamda1*Ladv
            print Loss.data.numpy()[0]
            Loss.backward()
            optimizer.step()
            
                
def build2pairs(train_data):
    data = []
    for i in range(min( len(train_data[0]), len(train_data[1]) )):
           data.append([train_data[0][i], train_data[1][i]])
    return data

def shuffleData(train_data):
    """
    this function don't need to return any value and the list is changed inplace
    """
    if len(train_data) == 2:
        random.shuffle(train_data[0])
        random.shuffle(train_data[1])
    else:
        random.shuffle(train_data)


