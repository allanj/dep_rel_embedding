# 
# @author: Allan
#

import torch
import torch.nn as nn
import torch.nn.functional as F



class DepLabeledGCN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gcn_hidden_dim = config.dep_hidden_dim
        self.num_gcn_layers = config.num_gcn_layers
        self.gcn_mlp_layers = config.gcn_mlp_layers
        # gcn layer
        self.layers = self.num_gcn_layers
        self.device = config.device
        self.mem_dim = self.gcn_hidden_dim

        self.gcn_drop = nn.Dropout(config.gcn_dropout).to(self.device)

        # gcn layer
        self.W = nn.ModuleList()


        for layer in range(self.layers):
            input_dim = config.dep_emb_size if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim).to(self.device))

        self.dep_emb = nn.Embedding(len(config.deplabels), config.dep_emb_size).to(config.device)

        # # output mlp layers
        # in_dim = config.hidden_dim
        # layers = [nn.Linear(in_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]
        # for _ in range(self.gcn_mlp_layers - 1):
        #     layers += [nn.Linear(self.gcn_hidden_dim, self.gcn_hidden_dim).to(self.device), nn.ReLU().to(self.device)]
        #
        # self.out_mlp = nn.Sequential(*layers).to(self.device)

        self.scorer = nn.Linear(config.dep_hidden_dim, len(config.deplabels)).to(self.device)

    def forward(self, adj_matrix, gcn_inputs):

        """

        :param gcn_inputs:
        :param word_seq_len:
        :param adj_matrix: should already contain the self loop
        :param dep_label_matrix:
        :return:
        """
        adj_matrix = adj_matrix.to(self.device)
        denom = adj_matrix.sum(2).unsqueeze(2) + 1
        gcn_inputs = self.dep_emb(gcn_inputs)  ## B
        for l in range(self.layers):
            Ax = adj_matrix.bmm(gcn_inputs)  ## N x N  times N x h  = Nxh
            AxW = self.W[l](Ax)   ## N x m
            AxW = AxW + self.W[l](gcn_inputs)  ## self loop  N x h
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW)

        output = self.scorer(gcn_inputs)
        return output


