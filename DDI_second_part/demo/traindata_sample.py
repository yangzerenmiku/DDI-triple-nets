import pandas as pd
import numpy as np
import torch
import json
import random

## p2d net
def create_p2d_edge_index(args):


    with open('dataset/drug_number2protein_number.json','r',encoding='utf-8') as file:
        strx = file.read()
        dict = json.loads(strx)

    key_list,value_list = [],[]
    for key in dict.keys():
        for value in dict[key]:
            key_list.append(int(key))
            value_list.append(int(value))
    key_list = [i+19081 for i in key_list]

    # 加入自环
    temp = [i for i in range(19081,19081+267)]
    value_list.extend(temp)
    key_list.extend(temp)

    protein_index = torch.LongTensor(value_list)
    CID_index = torch.LongTensor(key_list)
    pd_index = torch.stack((protein_index,CID_index),1)
    pd_index = pd_index.permute(1,0)
    return pd_index.to(args.device)

## ddi net
def create_ddi_edge_index(data,args,list=None):

    #### 生成直接 版本 邻接矩阵
    if list!=None:
        data = data.iloc[list]
        data.index = range(len(list))

    CID1,CID2 = data['drug1_order'],data['drug2_order']
    CID1 = torch.LongTensor(CID1)
    CID2 = torch.LongTensor(CID2)
    edge_idx = torch.stack((CID1,CID2))

    edge_weight = data['similarity']
    edge_weight = torch.LongTensor(edge_weight)
    return edge_idx.to(args.device),edge_weight.to(args.device)



# d2poly_d net

def create_d2poly_d_edge_index_and_label(data,args,list=None):

    if list!=None:
        data = data.iloc[list]
        data.index = range(len(list))

    CID1_order,CID2_order = data['drug1_order'],data['drug2_order']

    d_list,poly_d_list = [],[]
    for i in range(data.shape[0]):
        poly_d_list.append(i)
        poly_d_list.append(i)
        d_list.append(CID1_order[i])
        d_list.append(CID2_order[i])

    d_list = [i for i in d_list]
    poly_d_list = [i+267 for i in poly_d_list]

    # # 加入自环
    # temp = [i for i in range(548,data.shape[0]+548)]
    # d_list.extend(temp)
    # poly_d_list.extend(temp)

    d_list = torch.LongTensor(d_list)
    poly_d_list = torch.LongTensor(poly_d_list)
    d2poly_d_edge_index = torch.stack((d_list,poly_d_list),1)
    d2poly_d_edge_index = d2poly_d_edge_index.permute(1,0)

    label = data['label']
    label = [int(label[i]) for i in range(label.shape[0])]
    label = torch.LongTensor(label)
    return d2poly_d_edge_index.to(args.device),label.to(args.device)

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# protein2poly_d

def create_p2poly_d_edge_index(data,args,list=None):

    if list!=None:
        data = data.iloc[list]
        data.index = range(len(list))

    with open('dataset/drug_number2protein_number.json','r',encoding='utf-8') as file:
        _str = file.read()
        dict = json.loads(_str)

    CID1_order = data['drug1_order']
    CID2_order = data['drug2_order']
    poly_d_list , protein_list = [],[]

    for i in range(data.shape[0]):
        if str(CID1_order[i]) in dict:
            pro_temp_list = dict[str(CID1_order[i])]
            protein_list.extend(pro_temp_list)
            poly_temp_list = [i for x in pro_temp_list]
            poly_d_list.extend(poly_temp_list)

        if str(CID2_order[i]) in dict:
            pro_temp_list = dict[str(CID2_order[i])]
            protein_list.extend(pro_temp_list)
            poly_temp_list = [i for x in pro_temp_list]
            poly_d_list.extend(poly_temp_list)

    poly_d_list = [i+19081 for i in poly_d_list]

    # 加入自环
    temp = [i for i in range(19081,data.shape[0]+19081)]
    protein_list.extend(temp)
    poly_d_list.extend(temp)

    protein = torch.LongTensor(protein_list)
    poly_d = torch.LongTensor(poly_d_list)
    edge_index = torch.stack((protein,poly_d),0)
    return edge_index.to(args.device)

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

def train_sample(args,data):

    split_num = 0
    for i in range(data.shape[0]):
        if data['label'][i] > 0:
            split_num = i
            break
    assert split_num!=0
    data0 = [i for i in range(split_num)]
    train_label0_list = random.sample(data0,int(args.layer3_num_nodes/2))
    train_label0_list.sort()
    train_ddi0_list = [i for i in data0 if i not in train_label0_list]

    # train_ddi0_list =  random.sample(train_ddi0_list,70000)
    # train_ddi0_list.sort()

    data1 = [i for i in range(split_num, data.shape[0])]
    train_label1_list = random.sample(data1, int(args.layer3_num_nodes / 2))
    train_label1_list.sort()
    train_ddi1_list = [i for i in data1 if i not in train_label1_list]

    # train_ddi1_list =  random.sample(train_ddi1_list,70000)
    # train_ddi1_list.sort()

    ddi0_edge_index,ddi0_edge_weight = create_ddi_edge_index(data, args, train_ddi0_list)
    ddi1_edge_index,ddi1_edge_weight = create_ddi_edge_index(data, args, train_ddi1_list)

    train_ddi0_list.extend(train_ddi1_list)
    train_label0_list.extend(train_label1_list)

    p2poly_d_edge_index = create_p2poly_d_edge_index(data, args,train_label0_list)


    d2poly_d_edge_index,label = create_d2poly_d_edge_index_and_label(data,args,train_label0_list)

    return d2poly_d_edge_index,p2poly_d_edge_index,ddi0_edge_index,ddi0_edge_weight,ddi1_edge_index,ddi1_edge_weight ,label



