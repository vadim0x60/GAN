# coding: utf-8
import sys
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


def trainVAE_D(epoches,batch_size,data,ds_model,ds_emb,pretrainD=False):
	# load some necessary model and data
	gan = torch.load('./Model/gan.pkl')
	style = StyleData()
	style.load('./data/all_style')
	const = Constants(style.n_words)
	optimizer = optim.Adam(gan.parameters(),lr=const.Lr)
	lamda1 = 1
	lamda2 = 1
	lamda3 = 3
	cross_entropy = nn.CrossEntropyLoss()

	# init the state of some model
    ds_model.train(False)
    ds_emb.train(False)

    # prepare the train_data
    train_data = indexData2variable(data)

    # start the training loop
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

            # before we let the D lead the gradient the D model must be strong enough
            if random.choice([0,1]) and not pretrainD:
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


if __name__ == "__main__":
	"""
	you shuld use this script in this way:
	python trainVAE_D.py <epoches> <batch_size> <pretrainD?> <traindatafilename>

	for instance: 
	python trainVAE_D.py 1000 20 yes/no ./data/trainDataOfIndex.npy
	"""
	
	booldic = {'yes':True,
				'y':True,
				'Y':True,
				'Yes':True,
				'YES':True,
				'no':False,
				'N':False,
				'n':False,
				'NO':False,
				'No':False,}
	ds = torch.load('./Model/Ds.pkl')
	ds_emb = torch.load('./Model/embedding.pkl')
	train_data = np.load(sys.argv[4])
	epoches = int(sys.argv[1])
	batch_size = int(sys.argv[2])
	pretrainD = booldic[sys.argv[3]]

	trainVAE_D(epoches,batch_size,train_data,ds,ds_emb,pretrainD)