# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, TransformerConv,GATConv
from torch_geometric.nn.conv import  GCNConv, MessagePassing
from torch.nn import Parameter
import math
import numpy as np


class BaseGNN(torch.nn.Module):
    def __init__(self, dropout, num_layers):
        super(BaseGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t,edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            ######################自己加入的规范化###########################
            x = F.normalize(x)
            ######################自己加入的规范化###########################
        x = self.convs[-1](x, adj_t)
        if self.num_layers == 1:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(first_channels, second_channels))


class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(first_channels, second_channels, normalize=False))


class WSAGE(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(WSAGE, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GraphConv(first_channels, second_channels))


class Transformer(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(Transformer, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(TransformerConv(first_channels, second_channels))

class GAT(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GAT, self).__init__(dropout, num_layers)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GATConv(first_channels, second_channels))


class MLPPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class MLPCatPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPCatPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        in_channels = 2 * in_channels
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x1 = torch.cat([x_i, x_j], dim=-1)
        x2 = torch.cat([x_j, x_i], dim=-1)
        for lin in self.lins[:-1]:
            x1, x2 = lin(x1), lin(x2)
            x1, x2 = F.relu(x1), F.relu(x2)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.lins[-1](x1)
        x2 = self.lins[-1](x2)
        x = (x1 + x2)/2
        return x



class p2d_HierarchyConv(MessagePassing):

    def __init__(self,args):
        super(p2d_HierarchyConv, self).__init__(aggr='mean')
        self.layer1_num_nodes = args.layer1_num_nodes
        self.weight = Parameter(torch.Tensor(args.d_dim, args.p_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)

    def forward(self,x,edge_index):

        x = self.propagate(x=x, edge_index=edge_index)
        x = x[self.layer1_num_nodes:]
        x =  F.linear(x, self.weight)
        x = F.relu(x, inplace=True)
        return x



class d2poly_d_HierarchyConv(MessagePassing):

    def __init__(self,args):
        super(d2poly_d_HierarchyConv, self).__init__(aggr='mean')
        self.layer2_num_nodes = args.layer2_num_nodes
        self.weight = Parameter(torch.Tensor(args.poly_d_dim, args.d_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)


    def forward(self, x,edge_index):
        x = self.propagate(x=x, edge_index=edge_index)
        x = x[self.layer2_num_nodes:]
        x = F.linear(x, self.weight)
        x = F.relu(x, inplace=True)
        return x


class p2poly_d_HierarchyConv(MessagePassing):

    def __init__(self,args):
        super(p2poly_d_HierarchyConv, self).__init__(aggr='mean')
        self.layer1_num_nodes = args.layer1_num_nodes
        self.weight = Parameter(torch.Tensor(args.poly_d_dim, args.p_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)


    def forward(self, x,edge_index):
        x = self.propagate(x=x, edge_index=edge_index)
        x = x[self.layer1_num_nodes:]
        x = F.linear(x, self.weight)
        x = F.relu(x, inplace=True)
        return x




class LinearModel(torch.nn.Module):
    def __init__(self,args):
        super(LinearModel, self).__init__()
        self.lin1 = torch.nn.Linear(args.poly_d_dim*2 ,int(args.poly_d_dim/2))
        self.lin2 = torch.nn.Linear(int(args.poly_d_dim/2),2)

    def forward(self,poly_d_embedding):
        poly_d_embedding = self.lin1(poly_d_embedding)
        poly_d_embedding = F.relu(poly_d_embedding, inplace=True)
        poly_d_embedding = self.lin2(poly_d_embedding)
        return poly_d_embedding




class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(
            log_pred, target,reduction=self.reduction,ignore_index=self.ignore_index)


import torch
import torch.nn as nn


class LabelSmoothingRegularization(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        # labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')