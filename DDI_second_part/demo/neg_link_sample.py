# -*- coding: utf-8 -*-

import torch
from torch_geometric.utils import negative_sampling, add_self_loops


def neg_sample(edge_index, num_nodes,
                      num_neg, method='sparse'):

    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(new_edge_index, num_nodes=num_nodes,
                                 num_neg_samples=num_neg, method=method)

    neg_src = neg_edge[0]
    neg_dst = neg_edge[1]
    if neg_edge.size(1) < num_neg:
        k = num_neg - neg_edge.size(1)
        rand_index = torch.randperm(neg_edge.size(1))[:k]
        neg_src = torch.cat((neg_src, neg_src[rand_index]))
        neg_dst = torch.cat((neg_dst, neg_dst[rand_index]))

    return torch.reshape(torch.stack(
        (neg_src, neg_dst), dim=-1), (2, num_neg))

