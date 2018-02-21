# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

# there are four models need to be defined Ez, Ey, D, Ds(pre-trained)

class DsModel(nn.Module):
    """
    notes:
        This model can also be called classfier
    """

    def __init__(self, kind_filters, num_filters, num_in_channels, embedded_size, hidden_size=128):
        """
        Argus:
        kind_filters is a list
        num_filters is the number of filters we want use
        num_in_channels in this case is the number of kinds of embedding
        embedded_size is the embedding size (easy)
        hidden_size = is the hidden_units' number we want to use
        
        Notice:
        kind_filters need to be a list.
        for instance, [1, 2, 3] represent the there are three kind of
        window which's size is 1 or 2 or 3
        the Ds have multi-filter-size and muti-convs-maps
        """
        super(DsModel, self).__init__()

        self.kind_filters = kind_filters
        self.num_filters = num_filters

        self.convs = nn.ModuleList([])
        for width in self.kind_filters:
            self.convs.append(nn.Conv2d(num_in_channels, num_filters, (width, embedded_size)))

        self.linear = nn.Linear(num_filters * len(kind_filters), hidden_size)
        self.linear_out = nn.Linear(hidden_size, 2)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        this model's inputs should like this N_batch*C_channel*Seq_length*Embedded_size
        if we just use one kind embedding (dynamic or static) the C_channel is 1
        if we use two kind of embedding (dynamic and static) the C_channel is 2
        
        the outputs is the probability of x1 < X1
        """
        convs_outputs = []
        for convs in self.convs:
            convs_outputs.append(convs(x))

        max_pools_outputs = []
        for outputs in convs_outputs:
            max_pools_outputs.append(F.max_pool2d(outputs, kernel_size=(outputs.size()[2], 1)))
            # [2] is the size of high

        flatten = torch.cat(max_pools_outputs, dim=1).view(x.size()[0], -1)
        return self.softmax(self.relu(self.linear_out(self.drop(self.relu(self.linear(flatten))))))


class EzModel(nn.Module):
    """
    this model take embedding as the input
    this model is the decode model and it's hidden_output will be delivered to G model
    this model is implemented with a GRU(RNN) and the last hidden outputs as
    the encoded contents of input_sequence
    """

    def __init__(self, embedding_size, hidden_size):
        super(EzModel, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(embedding_size, hidden_size)
        # the GRU's output is special, 'hidden_out' of every time step excatly
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        """
        x should look like this shape seq_len * batch * input_size
        and as usual the batch is 1
        the output is the all hidden and the hidden is the last hidden
        """
        outputs, hidden = self.gru(x, hidden)
        return self.relu(outputs), self.relu(hidden)

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))  # the minibatch is 1


class EyModel(nn.Module):
    """
    this model is the style decode model who's output is deliverd to G model
    and this model's structure is also very similar with the Ds model
    """

    def __init__(self, in_channels, num_filters, kind_filters, embedding_size):
        super(EyModel, self).__init__()

        self.embedding_size = embedding_size
        self.kind_filters = kind_filters
        self.num_filters = num_filters
        self.y2_style = self.init_style()
        self.x2_style_flag = 1
        # we define the y2 style's flag is 1 and is x's index is 1 return y*

        self.convs = nn.ModuleList([])
        for width in kind_filters:
            self.convs.append(nn.Conv2d(in_channels, num_filters, (width, embedding_size)))

        self.relu = nn.ReLU()

    def forward(self, x, index):
        """
        this model's input should like this N_batch*C_channel*Len_seqence*Width_{embedding_size}
        and the input also should include the domain of x represent as {1,0} ,1 represent the 
        target domain
        and his output is regard as the syle represent (size: 1 * (Kinds_filters*Num_filters))
        """
        if index == self.x2_style_flag:
            return self.y2_style

        convs_outputs = []
        for convs in self.convs:
            convs_outputs.append(convs(x))

        max_pools_outputs = []
        for outputs in convs_outputs:
            max_pools_outputs.append(F.max_pool2d(outputs, kernel_size=(outputs.size()[2], 1)))
            
        y1_style = torch.cat(max_pools_outputs, dim=1).view(x.size()[0], -1)
        return self.relu(y1_style)

    def init_style(self):
        return nn.Parameter(torch.randn(1, len(self.kind_filters) * self.num_filters), requires_grad=True)


class GModel(nn.Module):
    def __init__(self, hidden_size, n_vocab, embedding_size, temper):  # temper is the temperature
        """
        Notice:
        GModel every time just take one word as input not a sequence
        the input should be one embedding, and the input_size should be the n_vocab
        the output_size should same with the n_vocabulary
        so the input_size == output_size
        """
        super(GModel, self).__init__()
        self.hidden_size = hidden_size
        self.temper = temper

        self.gru = nn.GRU(embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, n_vocab)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        """
        the input should look like this Len_sequence*N_batch*Embedding,the hidden seq == 1 other keep same
        N_batch is usually 1, and the Len_sequence is 1 and in this way can we collect the hidden
        the output is 1 * onehot_size
        """

        hidden, output = self.gru(x, hidden)
        output = self.softmax(self.relu(self.out(self.relu(hidden))).view(output.size()[0], -1))
        hidden = self.relu(hidden)

        logit = output
        return self.softmax(logit / self.temper), logit, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size), requires_grad=False)


class DModel(nn.Module):
    """
    DModel is also very like Ey & Ds and also have a sigmoid function as the output layer 
    and this model take G's hidden state and G's hidden state' length is dynamic
    """

    def __init__(self, kind_filters, num_filters, num_in_channels, width, hidden_size=128):

        super(DModel, self).__init__()
        self.kind_filters = kind_filters
        self.num_filters = num_filters

        self.convs = nn.ModuleList([])
        for w in self.kind_filters:
            self.convs.append(nn.Conv2d(num_in_channels, num_filters, (w, width)))

        self.linear = nn.Linear(num_filters * len(kind_filters), hidden_size)
        self.linear_out = nn.Linear(hidden_size, 2)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        the input is just the like this N_batch*C_channel*Seq_len*Width
        and the C_channel is 1 because the GRU only output one hidden state every time t
        """
        convs_outputs = []
        for convs in self.convs:
            convs_outputs.append(convs(x))

        max_pools_outputs = []
        for outputs in convs_outputs:
            max_pools_outputs.append(F.max_pool2d(outputs, kernel_size=(outputs.size()[2], 1)))

        flatten = torch.cat(max_pools_outputs, dim=1).view(x.size()[0], -1)
        return self.softmax(self.relu(self.linear_out(self.relu(self.drop(self.linear(flatten))))))


class Embed(nn.Module):
    """
    this is the embedding layer which could embed the index and one-hot logit vector
    but you should indicator use_one_hot or not with index = {True, False}
    """

    def __init__(self, n_vocab, embedding_size):
        super(Embed, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_size)

    def forward(self, x, index=True):
        if index:
            return self.embedding(x)
        else:
            return torch.mm(x, self.embedding.weight)


class GANModel(nn.Module):
    """
    this model is the gan model wich will reutrn many data we need to compute the loss so we just need
    create one model called GAN
    """

    def __init__(self, style_represent, content_represent, D_filters, D_num_filters, Ey_filters,
                 Ey_num_filters, embedding_size, n_vocab, temper, max_len=40):
        """
        style_represent is the dim we choose to represent the style
        content_represent is the dim we choose to represent the content
        D_filters is a list like this [1,2,3,4]
        D_num_filters is the the filters number we want to use for each window size
        Ey_filters
        """
        super(GANModel, self).__init__()
        self.style_represent = Ey_num_filters * len(Ey_filters)
        self.temper = temper
        self.max_len = max_len
        self.Ez = EzModel(embedding_size, content_represent)  # hidden_size is the content_represent
        # content_represent == conten_represent
        self.Ey = EyModel(1, Ey_num_filters, Ey_filters, embedding_size)
        # style_represent == Ey_num_filters * Len(Ey_filters)
        self.G = GModel(content_represent + self.style_represent, n_vocab, embedding_size, temper)
        self.D = DModel(D_filters, D_num_filters, 1, content_represent + self.style_represent)
        self.embedding = Embed(n_vocab, embedding_size)
        self.go = self.embedding(Variable(torch.LongTensor([0])))

    def forward(self, x1, x2, Ez_train=True,
                G_train=True,
                D_train=True,
                Embedd_train=True,
                Ey_train=True,
                Lcyc=True,
                Ladv=True,
                Ldis=True,
                Lrec=True):
        """
        the input x don't need to have a <go>, but must have an <EOS>
        The input x1, x2's shape should look like this (Len_seq)
        addition the N_batch must equal to 1 if we want to add batch training we can consider 
        implement outside the model because we need to use many middle output to compute the loss it will be 
        very complicated if we compute inside the model
        Notice:
        there is something we need to pay attention to is will the y_start will be changed, if 
        y_start is not be changed we need to consider a method to update the y_star
        """
        self.Ez.train(Ez_train)
        self.Ey.train(Ey_train)
        self.D.train(D_train)
        self.G.train(G_train)
        self.embedding.train(Embedd_train)

        x1_hat = 0
        x1_hat_noT = 0
        x1_hat_hid = 0
        x2_hat = 0
        x2_hat_noT = 0
        x2_hat_hid = 0
        y1 = 0
        y_star = 0
        x1_wl_hid = 0
        x1_bar = 0
        x1_bar_noT = 0
        x2_bar = 0
        x2_bar_noT = 0
        D_x1_wl = 0
        D_x2_hat = 0

        # x1 and x2 is index represent
        embedd_x1 = self.embedding(x1)
        embedd_x2 = self.embedding(x2)
        y1 = self.Ey(embedd_x1.unsqueeze(0).unsqueeze(0), 0)  # we need to shape the 2d variable to 4d variable
        y_star = self.Ey.y2_style
        # and now y1's shape is 1*m

        hidden = self.Ez.init_hidden()

        if Lrec or Lcyc or Ladv:
            outputs, z1 = self.Ez(embedd_x1.unsqueeze(1), hidden)
            outputs, z2 = self.Ez(embedd_x2.unsqueeze(1), hidden)
            # and the z1, z2's shape is 1 * 1 * hidden_size

        x1_seq_len = x1.size()[0]
        x2_seq_len = x2.size()[0]

        if Lrec or Ladv:
            x1_hat, x1_hat_noT, x1_hat_hid = self.get_x_hat_hidden(z1, y1, x1_seq_len)
            # x1_hat_hid is of no use
            x2_hat, x2_hat_noT, x2_hat_hid = self.get_x_hat_hidden(z2, y_star, x2_seq_len)

        if Lcyc or Ladv:
            x1_wl, x1_wl_noT, x1_wl_hid = self.get_x_hat_hidden(z1, y_star, self.max_len, length_fix=False)
            if Lcyc:
                x2_wl, x2_wl_noT, x2_wl_hid = self.get_x_hat_hidden(z2, y1, self.max_len, length_fix=False)

        if Lcyc:
            embedd_x1_wl = self.embedding(x1_wl, index=False)
            outputs, z1_wl = self.Ez(embedd_x1_wl.unsqueeze(1), hidden)
            x1_bar, x1_bar_noT, x1_bar_hid = self.get_x_hat_hidden(z1_wl, y1, x1_seq_len)

            embedd_x2_wl = self.embedding(x2_wl, index=False)
            outputs, z2_wl = self.Ez(embedd_x2_wl.unsqueeze(1), hidden)
            x2_bar, x2_bar_noT, x2_bar_hid = self.get_x_hat_hidden(z2_wl, y_star, x2_seq_len)
        if Ladv:
            D_x1_wl = self.D(x1_wl_hid.view(1, 1, x1_wl_hid.size()[0], -1))
            D_x2_hat = self.D(x2_hat_hid.view(1, 1, x2_hat_hid.size()[0], -1))
        # one question, do we need to return x1_hat and x2_hat x1_bar and x2_bar without enforcement?

        return {'x1_hat': x1_hat,
                'x1_hat_hid': x1_hat_hid,
                'x1_hat_noT': x1_hat_noT,
                'x2_hat': x2_hat,
                'x2_hat_hid': x2_hat_hid,
                'x2_hat_noT': x2_hat_noT,
                'y1': y1,
                'y_star': y_star,
                'x1_wl_hid': x1_wl_hid,
                'x1_bar': x1_bar,
                'x1_bar_noT': x1_bar_noT,
                'x2_bar': x2_bar,
                'x2_bar_noT': x2_bar_noT,
                'D_x1_wl': D_x1_wl,
                'D_x2_hat': D_x2_hat}

    def get_x_hat_hidden(self, z, y, seq_len, length_fix=True):
        x_hats = []
        x_hats_noT = []
        hiddens = []
        x_hat, x_hat_noT, hidden = self.G(self.go.view(1, 1, -1),
                                          torch.cat([z.view(1, -1), y], dim=-1).view(1, 1, -1))
        x_hats.append(x_hat)
        x_hats_noT.append(x_hat_noT)
        hiddens.append(hidden)
        for i in range(1, seq_len):

            embedd_x_hat = self.embedding(x_hat, index=False)
            x_hat, x_hat_noT, hidden = self.G(embedd_x_hat.view(1, 1, -1), hidden.view(1, 1, -1))

            # the sequence's length be generated should be larger than 6 at least 
            if x_hat.topk(1)[1].data.numpy() == 1 and not length_fix and i >= 6:  # 1 represent the end of the seq
                break

            x_hats.append(x_hat)
            x_hats_noT.append(x_hat_noT)
            hiddens.append(hidden)

        return torch.cat(x_hats), torch.cat(x_hats_noT), torch.cat(hiddens)  # cat in the first dim
