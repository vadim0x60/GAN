
# coding: utf-8
# some parametes should be adjusted in here
class Constants(object):
    def __init__(self, n_vocab):
        self.Lr = 0.0001
        self.Embedding_size = 250
        self.Content_represent = 250
        self.Style_represent = 500
        self.Ey_filters = [1, 2, 3, 4, 5]
        self.Ey_num_filters = 100
        self.D_filters = [2, 3, 4, 5, 6]
        self.D_num_filters = 100
        self.Ds_filters = [1, 2, 3, 4]
        self.Ds_num_filters = 100
        self.Hidden_size = 248
        self.N_vocab = n_vocab
        self.Temper = 0.0001
        self.Max_len = 40
        self.Min_len = 6 # 6 is the max window size of the filters
        self.use_cuda = True
        

