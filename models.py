# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

class TEM(torch.nn.Module):
    def __init__(self, opt):
        super(TEM, self).__init__()
        
        self.feat_dim = opt["tem_feat_dim"]
        self.temporal_dim = opt["temporal_scale"]
        self.batch_size= opt["tem_batch_size"]
        self.c_hidden = opt["tem_hidden_dim"]
        self.c_hidden1 = opt["tem_hidden_dim1"]
        self.c_hidden2 = opt["tem_hidden_dim2"]
        self.r_dropout = opt["tem_dropout"]
        self.n_layer = opt["tem_layer"]
        self.rnn_type = opt["rnn_type"]
        self.tem_best_loss = 10000000
        self.output_dim = 3
        
        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim,out_channels=self.c_hidden,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden,out_channels=self.c_hidden,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.c_hidden,out_channels=self.output_dim,kernel_size=1,stride=1,padding=0)
        self.reset_params()

        self.lstm1 = torch.nn.LSTM(input_size=self.feat_dim, hidden_size=self.c_hidden1, num_layers=self.n_layer,
                                   batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(p=self.r_dropout)
        self.fc = torch.nn.Linear(in_features=self.c_hidden,out_features=self.output_dim)
        self.lstm2 = torch.nn.LSTM(input_size=self.c_hidden2, hidden_size=self.output_dim,
                                   batch_first=True, bidirectional=True)

        self.rnn = getattr(nn, self.rnn_type)(self.feat_dim, self.c_hidden, self.n_layer, batch_first=True, dropout=self.r_dropout)
        self.scores = torch.nn.Linear(self.c_hidden, self.output_dim)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(0.01 * self.conv3(x))
        return x

    # def forward(self, x):
    #     x, _ = self.lstm1(x)
    #     x, _ = self.lstm2(x)
    #     x = (x+1)/2
    #     x = torch.transpose(x,1,2)
    #     fw = x[:,:self.output_dim,:]
    #     bw = x[:,self.output_dim:,:]
    #     x =(fw+bw)/2
    #     return x

    # def forward(self, features):
    #     N, T, _ = features.size()
    #     rnn_output, _ = self.rnn(features)
    #     rnn_output = rnn_output.contiguous()
    #     rnn_output = rnn_output.view(rnn_output.size(0) * rnn_output.size(1), rnn_output.size(2))
    #     outputs = torch.sigmoid(self.scores(rnn_output))
    #     outputs = outputs.view(N, T, self.output_dim)
    #     outputs = torch.transpose(outputs,1,2)
    #     return outputs

class PEM(torch.nn.Module):
    
    def __init__(self,opt):
        super(PEM, self).__init__()
        
        self.feat_dim = opt["pem_feat_dim"]
        self.batch_size = opt["pem_batch_size"]
        self.hidden_dim = opt["pem_hidden_dim"]
        self.u_ratio_m = opt["pem_u_ratio_m"]
        self.u_ratio_l = opt["pem_u_ratio_l"]
        self.output_dim = 1
        self.pem_best_loss = 1000000
        
        self.fc1 = torch.nn.Linear(in_features=self.feat_dim,out_features=self.hidden_dim,bias =True)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_dim,out_features=self.output_dim,bias =True)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(0.1*self.fc1(x))
        x = torch.sigmoid(0.1*self.fc2(x))
        return x
