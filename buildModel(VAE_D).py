# coding: utf-8
import numpy as np
import random
import torch
from Constant import Constants
from ModelDefine import GANModel
from load_data import StyleData
from PreTrainDs import indexData2variable





if __name__ == "__main__":
    # this is just to build the gan model and save the network to use later
    style = StyleData()
    style.load('./traindata/style')
    const = Constants(n_vocab=style.n_words)
    gan = GANModel(content_represent=const.Content_represent,
                   D_filters=const.D_filters,
                   D_num_filters=const.Ds_num_filters,
                   embedding_size=const.Embedding_size,
                   Ey_filters=const.Ey_filters,
                   Ey_num_filters=const.Ey_num_filters,
                   n_vocab=const.N_vocab,
                   style_represent=const.Style_represent,
                   temper=const.Temper)  # there are 9 parameters of a GAN
    torch.save(gan, './Model/gan.pkl')
    print 'finished'
