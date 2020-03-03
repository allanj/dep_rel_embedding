import torch
import torch.nn as nn
from torch.nn import Parameter
from enum import IntEnum
import math
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class NaiveLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # Define/initialize all tensors
        # forget gate
        self.Wf = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bf = Parameter(torch.Tensor(hidden_sz))
        # input gate
        self.Wi = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bi = Parameter(torch.Tensor(hidden_sz))
        # Candidate memory cell
        self.Wc = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bc = Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.Wo = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bo = Parameter(torch.Tensor(hidden_sz))
        self.init_weights()
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
        # for p in self.parameters():
        #     if p.data.ndimension() >= 2:
        #         nn.init.xavier_uniform_(p.data)
        #     else:
        #         nn.init.zeros_(p.data)
    # Define forward pass through all LSTM cells across all timesteps.
    # By using PyTorch functions, we get backpropagation for free.
    def forward(self, x: torch.Tensor,
                init_states = None
                ):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        # ht and Ct start as the previous states and end as the output states in each loop bellow
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            # print(xt.size())
            hx_concat = torch.cat((ht, xt), dim=1)
            ### The LSTM Cell!
            ft = torch.sigmoid(hx_concat @ self.Wf + self.bf)
            it = torch.sigmoid(hx_concat @ self.Wi + self.bi)
            Ct_candidate = torch.tanh(hx_concat @ self.Wc + self.bc)
            ot = torch.sigmoid(hx_concat @ self.Wo + self.bo)
            # outputs
            Ct = ft * Ct + it * Ct_candidate
            ht = ot * torch.tanh(Ct)
            hidden_seq.append(ht.unsqueeze(Dim.batch))
            if t == 0:
                ht = ht
            else:
                ht = torch.max(torch.stack(hidden_seq), dim=0)[0].squeeze()
                # print(Ct.size())
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (ht, Ct)

