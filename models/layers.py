import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, ops, adj):
        ops = F.dropout(ops, self.dropout, self.training)
        support = F.linear(ops, self.weight)
        output = F.relu(torch.matmul(adj, support))

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + '->' + str(self.out_features) + ')'
