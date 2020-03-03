import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):

    def __init__(self,
                 config):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.device = config.device
        self.dep_label_embedding = nn.Embedding(len(config.deplabel2idx), config.dep_emb_size).to(self.device)
        # input layer
        self.layers.append(GraphConv(config.dep_emb_size, config.dep_hidden_dim, activation=nn.ReLU()))
        # hidden layers
        for i in range(config.gcn_mlp_layers - 1):
            self.layers.append(GraphConv(config.dep_hidden_dim, config.dep_hidden_dim, activation=nn.ReLU()))
        num_dep_labels = len(config.deplabels)
        # output layer
        self.layers.append(GraphConv(config.dep_hidden_dim, num_dep_labels))
        self.dropout = nn.Dropout(p=config.gcn_dropout)

    def forward(self, g, dep_labels):
        features = self.dep_label_embedding(dep_labels)
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h