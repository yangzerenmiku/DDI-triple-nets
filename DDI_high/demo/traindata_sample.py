import pandas as pd
import numpy as np
import torch
import json
import random



# d2poly_d net

def create_d2poly_d_edge_index(data,args):

    CID1_order = data['CID1_order']
    CID2_order = data['CID2_order']
    CID3_order = data['CID3_order']
    d_list,poly_d_list = [],[]
    for i in range(data.shape[0]):
        poly_d_list.append(i)
        poly_d_list.append(i)
        poly_d_list.append(i)
        d_list.append(CID1_order[i])
        d_list.append(CID2_order[i])
        d_list.append(CID3_order[i])

    d_list = [i for i in d_list]
    poly_d_list = [i+195 for i in poly_d_list]

    # 加入自环
    temp = [i for i in range(195,data.shape[0]+195)]
    d_list.extend(temp)
    poly_d_list.extend(temp)

    d_list = torch.LongTensor(d_list)
    poly_d_list = torch.LongTensor(poly_d_list)
    d2poly_d_edge_index = torch.stack((d_list,poly_d_list),1)
    d2poly_d_edge_index = d2poly_d_edge_index.permute(1,0)
    return d2poly_d_edge_index.to(args.device)

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# protein2poly_d

def create_p2poly_d_edge_index(data,args):
    with open('dataset/CID_order2protein_order.json','r',encoding='utf-8') as file:
        _str = file.read()
        dict = json.loads(_str)

    CID1_order = data['CID1_order']
    CID2_order = data['CID2_order']
    CID3_order = data['CID3_order']
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

        if str(CID3_order[i]) in dict:
            pro_temp_list = dict[str(CID3_order[i])]
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

# train label

def create_smooth_label(data,args):

    OR = data['Odds Ratio']
    OR = OR.tolist()

    smooth_label = np.zeros((len(OR),2))
    eps = 0.01

    for i in range(len(OR)):
        if OR[i] < 1:
            smooth_label[i,0] = 1 - eps
            smooth_label[i,1] = eps
        elif OR[i] < 16:
            smooth_label[i,0] = -eps+0.2/i
            smooth_label[i,1] = eps+1-0.2/i
        else:
            smooth_label[i,0] = eps
            smooth_label[i,1] = 1-eps
    smooth_label = torch.from_numpy(smooth_label)


    return smooth_label.to(args.device)

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


def create_label(data,args):

    OR = data['Odds Ratio']
    OR = OR.tolist()
    OR = [0 if i<2 else 1 for i in OR]
    label = torch.LongTensor(OR)
    return label.to(args.device)

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

def train_sample(args,data):

    split_num = 0
    for i in range(data.shape[0]):
        if data['Odds Ratio'][i] >2:
            split_num = i
            break
    assert split_num!=0
    label0 = [i for i in range(split_num)]
    label1 = [i for i in range(split_num,data.shape[0])]
    train_label0_list = random.sample(label0,int(args.layer3_num_nodes/2))
    train_label0_list.sort()
    train_label1_list = random.sample(label1,int(args.layer3_num_nodes/2))
    train_label1_list.sort()
    train_label0_list.extend(train_label1_list)

    # x = data[0:0]
    # for m in train_label0_list:
    #     x = pd.concat((x,data[m:m+1]),axis=0)

    data = data.iloc[train_label0_list]
    data.index = range(len(train_label0_list))

    d2poly_d_edge_index = create_d2poly_d_edge_index(data,args)
    p2poly_d_edge_index = create_p2poly_d_edge_index(data,args)
    smooth_label = create_smooth_label(data,args)
    label = create_label(data, args)


    return d2poly_d_edge_index,p2poly_d_edge_index,smooth_label,label



