import pandas as pd
import numpy as np
import torch
import json
import scipy.sparse as sp

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

## ppi

def create_ppi_edge_index():

    #### 生成torch.sparse_coo_tensor 版本 邻接矩阵
    # data = pd.read_csv(r'ppi.csv',encoding='utf-8')
    # Gene_1_order = data['Gene 1_order']
    # Gene_2_order = data['Gene 2_oredr']
    # num_node = max(np.max(Gene_1_order),np.max(Gene_2_order))+1
    # adj = np.zeros((num_node,num_node))
    # for i in range(num_node):
    #     adj[Gene_1_order[i],Gene_2_order[i]] = 1
    #     adj[Gene_2_order[i],Gene_1_order[i]] = 1
    # adj = adj + np.eye(num_node)
    #
    # # a_matrix 是一个邻接矩阵
    # tmp_coo=sp.coo_matrix(adj)
    # values=tmp_coo.data
    # indices=np.vstack((tmp_coo.row,tmp_coo.col))
    # i=torch.LongTensor(indices)
    # v=torch.LongTensor(values)
    # edge_idx=torch.sparse_coo_tensor(i,v,tmp_coo.shape)

    #### 生成直接 版本 邻接矩阵，没加自环
    data = pd.read_csv(r'ppi.csv', encoding='utf-8')
    Gene_1_order = data['Gene 1_order']
    Gene_2_order = data['Gene 2_order']
    Gene_1_order = torch.from_numpy(np.array(Gene_1_order))
    Gene_2_order = torch.from_numpy(np.array(Gene_2_order))
    edge_idx1 = torch.cat((Gene_1_order,Gene_2_order))
    edge_idx2 = torch.cat((Gene_2_order,Gene_1_order))
    edge_idx = torch.stack((edge_idx1,edge_idx2),1)
    edge_idx = edge_idx.permute(1, 0)

    # 加入自环
    self_loop = torch.LongTensor([[i for i in range(19081)],[i for i in range(19081)]])
    edge_idx = torch.cat((edge_idx,self_loop),1)
    return edge_idx


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

#  pd-net

def create_p2d_edge_index():
    import json
    with open('drug_number2protein.json','r',encoding='utf-8') as file:
        strx = file.read()
        dict = json.loads(strx)

    key_list,value_list = [],[]
    for key in dict.keys():
        for value in dict[key]:
            key_list.append(int(key))
            value_list.append(int(value))
    key_list = [i+19081 for i in key_list]

    # 加入自环
    temp = [i for i in range(19081,19081+548)]
    value_list.extend(temp)
    key_list.extend(temp)

    protein_index = torch.LongTensor(value_list)
    CID_index = torch.LongTensor(key_list)
    pd_index = torch.stack((protein_index,CID_index),1)
    pd_index = pd_index.permute(1,0)
    return pd_index

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

#  ddi

def create_ddi_edge_index():

    #### 生成torch.sparse_coo_tensor 版本 邻接矩阵
    # data = pd.read_csv(r'2_last.csv',encoding='utf-8')
    # CID1 = data['CID1_order']
    # CID2 = data['CID2_order']
    # CID1 = np.array(CID1)
    # CID2 = np.array(CID2)
    # num_node = max(np.max(CID1),np.max(CID2))+1
    #
    # adj = np.zeros((num_node,num_node))
    # for i in range(data.shape[0]):
    #     adj[CID1[i],CID2[i]] = 1
    #     adj[CID2[i],CID1[i]] = 1
    # adj = adj + np.eye(num_node)
    #
    # # a_matrix 是一个邻接矩阵
    # tmp_coo=sp.coo_matrix(adj)
    # values=tmp_coo.data
    # indices=np.vstack((tmp_coo.row,tmp_coo.col))
    # i=torch.LongTensor(indices)
    # v=torch.LongTensor(values)
    # edge_idx=torch.sparse_coo_tensor(i,v,tmp_coo.shape)

    #### 生成直接 版本 邻接矩阵
    data = pd.read_csv(r'2_last.csv', encoding='utf-8')
    data = data[:3849]
    CID1,CID2 = data['CID1_order'],data['CID2_order']
    CID1 = CID1.tolist()
    CID2 = CID2.tolist()
    CID1 = [i for i in CID1]
    CID2 = [i for i in CID2]

    # 加入自环
    self_loop = [i for i in range(195)]
    self_loop = torch.LongTensor(self_loop)

    CID1 = torch.LongTensor(CID1)
    CID2 = torch.LongTensor(CID2)
    edge_idx1 = torch.cat((CID1, CID2,self_loop))
    edge_idx2 = torch.cat((CID2, CID1,self_loop))
    pos_edge_idx = torch.stack((edge_idx1,edge_idx2))

    data = pd.read_csv(r'2_last.csv', encoding='utf-8')
    data = data[3849:]
    CID1, CID2 = data['CID1_order'], data['CID2_order']
    CID1 = CID1.tolist()
    CID2 = CID2.tolist()
    CID1 = [i for i in CID1]
    CID2 = [i for i in CID2]

    # 加入自环
    self_loop = [i for i in range(195)]
    self_loop = torch.LongTensor(self_loop)

    CID1 = torch.LongTensor(CID1)
    CID2 = torch.LongTensor(CID2)
    edge_idx1 = torch.cat((CID1, CID2, self_loop))
    edge_idx2 = torch.cat((CID2, CID1, self_loop))
    neg_edge_idx = torch.stack((edge_idx1, edge_idx2))
    return pos_edge_idx,neg_edge_idx
    # 7893 3523

def create_ddi_edge_weight():
    data = pd.read_csv(r'2_last.csv',encoding='utf-8')
    OR = data['Odds Ratio']
    OR1 = OR[:3849]
    OR0 = OR[3849:]

    OR0 = 1/OR0

    OR1 = torch.LongTensor(OR1.tolist())
    pos_edge_weight = torch.cat((OR1,OR1,torch.LongTensor([i for i in range(195)])))

    OR0 = torch.LongTensor(OR0.tolist())
    neg_edge_weight = torch.cat((OR0, OR0, torch.LongTensor([i for i in range(195)])))

    return pos_edge_weight,neg_edge_weight
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# d2poly_d net

def create_train_d2poly_d_edge_index():

    data = pd.read_csv(r'3_last_train.csv',encoding='utf-8')
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


    d_list = torch.LongTensor(d_list)
    poly_d_list = torch.LongTensor(poly_d_list)
    d2poly_d_edge_index = torch.stack((d_list,poly_d_list),1)
    d2poly_d_edge_index = d2poly_d_edge_index.permute(1,0)
    return d2poly_d_edge_index

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# protein2poly_d

def create_train_protein2poly_d_edge_index():
    with open('CID_order2protein_order.json','r',encoding='utf-8') as file:
        _str = file.read()
        dict = json.loads(_str)

    data = pd.read_csv("3_last_train.csv")
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

    protein = torch.LongTensor(protein_list)
    poly_d_list = [i+19081 for i in poly_d_list]
    poly_d = torch.LongTensor(poly_d_list)
    edge_index = torch.stack((protein,poly_d),0)
    return edge_index


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
# poly_ddi

def create_train_poly_ddi_edge_index():

    data = pd.read_csv(r'3_last_train.csv',encoding='utf-8')
    list = []
    for i in range(data.shape[0]):
        temp = []
        temp.append(int(data['CID1_order'][i]))
        temp.append(int(data['CID2_order'][i]))
        temp.append(int(data['CID3_order'][i]))
        list.append(temp)

    numbers = []
    for i in range(len(list)):
        temp = list[i]
        number = []
        for j in range(len(list)):
            nums = 0
            if j!=i:
                if temp[0] in list[j]:
                    nums = nums + 1
                if temp[1] in list[j]:
                    nums = nums + 1
                if temp[2] in list[j]:
                    nums = nums + 1
                if nums >=2:
                    number.append(j)
        numbers.append(number)


    # a = np.zeros((len(numbers),1))
    # a = pd.DataFrame(a)
    # for i in range(len(numbers)):
    #     a[i] = str(numbers[i])
    # a.to_csv("x.csv")

    x = []
    for i in range(len(numbers)):
        temp = []
        for j in range(len(numbers[i])):
            temp.append(i)
        x.append(temp)

    index1,index2 = [],[]
    for i in range(len(x)):
        for j in range(len(x[i])):
            index1.append(x[i][j])
            index2.append(numbers[i][j])

    # index1 = [i+195 for i in index1]
    # index2 = [i+195 for i in index2]
    #### 生成torch.sparse_coo_tensor 版本 邻接矩阵
    # index1 = np.array(index1)
    # index2 = np.array(index2)
    # num_node = max(np.max(index1), np.max(index2)) + 1
    # adj = np.zeros((num_node,num_node))
    # for i in range(data.shape[0]):
    #     adj[index1[i],index2[i]] = 1
    #     adj[index2[i],index1[i]] = 1
    # adj = adj + np.eye(num_node)
    #
    # tmp_coo=sp.coo_matrix(adj)
    # values=tmp_coo.data
    # indices=np.vstack((tmp_coo.row,tmp_coo.col))
    # i=torch.LongTensor(indices)
    # v=torch.LongTensor(values)
    # edge_idx=torch.sparse_coo_tensor(i,v,tmp_coo.shape)

    #### 生成直接 版本 邻接矩阵,没加自环
    index1 = torch.LongTensor(index1)
    index2 = torch.LongTensor(index2)
    edge_idx1 = torch.cat((index1,index2))
    edge_idx2 = torch.cat((index2,index1))
    edge_idx = torch.stack((edge_idx1,edge_idx2),1)
    edge_idx = edge_idx.permute(1,0)
    return edge_idx



def create_train_poly_ddi_edge_index_OR():

    data = pd.read_csv(r'3_last_train.csv', encoding='utf-8')
    OR = data['Odds Ratio']
    OR = torch.from_numpy(np.array(OR))

    return OR




##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# label

def create_train_set_label():

    data = pd.read_csv(r'3_last_train.csv',encoding='utf-8')
    label = data['Odds Ratio']
    label = label.tolist()
    label = [0 if i < 2 else 1 for i in label]
    # x = [2 for i in range(195)]
    # x.extend(label)
    label = torch.LongTensor(label)
    return label

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# d2poly_d net

def create_test_d2poly_d_edge_index():
    data = pd.read_csv(r'3_last_test.csv',encoding='utf-8')
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
    d_list = torch.LongTensor(d_list)
    poly_d_list = torch.LongTensor(poly_d_list)
    d2poly_d_edge_index = torch.stack((d_list,poly_d_list),1)
    d2poly_d_edge_index = d2poly_d_edge_index.permute(1,0)
    return d2poly_d_edge_index

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# protein2poly_d

def create_test_p2poly_d_edge_index():
    with open('CID_order2protein_order.json','r',encoding='utf-8') as file:
        _str = file.read()
        dict = json.loads(_str)

    data = pd.read_csv("3_last_test.csv")
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

    protein = torch.LongTensor(protein_list)
    poly_d_list = [i+19081 for i in poly_d_list]
    poly_d = torch.LongTensor(poly_d_list)
    edge_index = torch.stack((protein,poly_d),0)
    return edge_index
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# poly_ddi
def create_test_poly_ddi_edge_index():
    data = pd.read_csv(r'3_last_test.csv',encoding='utf-8')
    list = []
    for i in range(data.shape[0]):
        temp = []
        temp.append(int(data['CID1_order'][i]))
        temp.append(int(data['CID2_order'][i]))
        temp.append(int(data['CID3_order'][i]))
        list.append(temp)

    numbers = []
    for i in range(len(list)):
        temp = list[i]
        number = []
        for j in range(len(list)):
            nums = 0
            if j!=i:
                if temp[0] in list[j]:
                    nums = nums + 1
                if temp[1] in list[j]:
                    nums = nums + 1
                if temp[2] in list[j]:
                    nums = nums + 1
                if nums >=2:
                    number.append(j)
        numbers.append(number)


    # a = np.zeros((len(numbers),1))
    # a = pd.DataFrame(a)
    # for i in range(len(numbers)):
    #     a[i] = str(numbers[i])
    # a.to_csv("x.csv")

    x = []
    for i in range(len(numbers)):
        temp = []
        for j in range(len(numbers[i])):
            temp.append(i)
        x.append(temp)

    index1,index2 = [],[]
    for i in range(len(x)):
        for j in range(len(x[i])):
            index1.append(x[i][j])
            index2.append(numbers[i][j])

    # index1 = [i+195 for i in index1]
    # index2 = [i+195 for i in index2]
    #### 生成torch.sparse_coo_tensor 版本 邻接矩阵
    # index1 = np.array(index1)
    # index2 = np.array(index2)
    # num_node = max(np.max(index1), np.max(index2)) + 1
    # adj = np.zeros((num_node,num_node))
    # for i in range(data.shape[0]):
    #     adj[index1[i],index2[i]] = 1
    #     adj[index2[i],index1[i]] = 1
    # adj = adj + np.eye(num_node)
    #
    # tmp_coo=sp.coo_matrix(adj)
    # values=tmp_coo.data
    # indices=np.vstack((tmp_coo.row,tmp_coo.col))
    # i=torch.LongTensor(indices)
    # v=torch.LongTensor(values)
    # edge_idx=torch.sparse_coo_tensor(i,v,tmp_coo.shape)

    #### 生成直接 版本 邻接矩阵
    # 加入自环
    self_loop = [i for i in range(data.shape[0])]
    index1.extend(self_loop)
    index2.extend(self_loop)

    index1 = torch.LongTensor(index1)
    index2 = torch.LongTensor(index2)
    edge_idx1 = torch.cat((index1,index2))
    edge_idx2 = torch.cat((index2,index1))
    edge_idx = torch.stack((edge_idx1,edge_idx2),1)
    edge_idx = edge_idx.permute(1,0)
    return edge_idx

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# label

def create_test_set_label():
    data = pd.read_csv(r'3_last_test.csv',encoding='utf-8')
    label = data['Odds Ratio']
    label = label.tolist()
    label = [0 if i < 2 else 1 for i in label]
    label = torch.LongTensor(label)
    return label

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


#****************************************************************

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# d2poly_d net

def create_4test_d2poly_d_edge_index():
    data = pd.read_csv(r'4_last.csv',encoding='utf-8')
    CID1_order = data['CID1_order']
    CID2_order = data['CID2_order']
    CID3_order = data['CID3_order']
    CID4_order = data['CID4_order']
    d_list,poly_d_list = [],[]
    for i in range(data.shape[0]):
        poly_d_list.append(i)
        poly_d_list.append(i)
        poly_d_list.append(i)
        poly_d_list.append(i)
        d_list.append(CID1_order[i])
        d_list.append(CID2_order[i])
        d_list.append(CID3_order[i])
        d_list.append(CID4_order[i])

    d_list = [i for i in d_list]
    poly_d_list = [i+195 for i in poly_d_list]
    d_list = torch.LongTensor(d_list)
    poly_d_list = torch.LongTensor(poly_d_list)
    d2poly_d_edge_index = torch.stack((d_list,poly_d_list),1)
    d2poly_d_edge_index = d2poly_d_edge_index.permute(1,0)
    return d2poly_d_edge_index

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# protein2poly_d

def create_4test_p2poly_d_edge_index():
    with open('CID_order2protein_order.json','r',encoding='utf-8') as file:
        _str = file.read()
        dict = json.loads(_str)

    data = pd.read_csv("4_last.csv")
    CID1_order = data['CID1_order']
    CID2_order = data['CID2_order']
    CID3_order = data['CID3_order']
    CID4_order = data['CID4_order']
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

        if str(CID4_order[i]) in dict:
            pro_temp_list = dict[str(CID4_order[i])]
            protein_list.extend(pro_temp_list)
            poly_temp_list = [i for x in pro_temp_list]
            poly_d_list.extend(poly_temp_list)

    protein = torch.LongTensor(protein_list)
    poly_d_list = [i+19081 for i in poly_d_list]
    poly_d = torch.LongTensor(poly_d_list)
    edge_index = torch.stack((protein,poly_d),0)
    return edge_index

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# label

def create_4test_set_label():
    data = pd.read_csv(r'4_last.csv',encoding='utf-8')
    label = data['Odds Ratio']
    label = label.tolist()
    label = [0 if i < 2 else 1 for i in label]
    label = torch.LongTensor(label)
    return label

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


#****************************************************************

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# d2poly_d net

def create_5test_d2poly_d_edge_index():
    data = pd.read_csv(r'5_last.csv',encoding='utf-8')
    CID1_order = data['CID1_order']
    CID2_order = data['CID2_order']
    CID3_order = data['CID3_order']
    CID4_order = data['CID4_order']
    CID5_order = data['CID5_order']
    d_list,poly_d_list = [],[]
    for i in range(data.shape[0]):
        poly_d_list.append(i)
        poly_d_list.append(i)
        poly_d_list.append(i)
        poly_d_list.append(i)
        poly_d_list.append(i)
        d_list.append(CID1_order[i])
        d_list.append(CID2_order[i])
        d_list.append(CID3_order[i])
        d_list.append(CID4_order[i])
        d_list.append(CID5_order[i])

    d_list = [i for i in d_list]
    poly_d_list = [i+195 for i in poly_d_list]
    d_list = torch.LongTensor(d_list)
    poly_d_list = torch.LongTensor(poly_d_list)
    d2poly_d_edge_index = torch.stack((d_list,poly_d_list),1)
    d2poly_d_edge_index = d2poly_d_edge_index.permute(1,0)
    return d2poly_d_edge_index

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# protein2poly_d

def create_5test_p2poly_d_edge_index():
    with open('CID_order2protein_order.json','r',encoding='utf-8') as file:
        _str = file.read()
        dict = json.loads(_str)

    data = pd.read_csv("5_last.csv")
    CID1_order = data['CID1_order']
    CID2_order = data['CID2_order']
    CID3_order = data['CID3_order']
    CID4_order = data['CID4_order']
    CID5_order = data['CID5_order']
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

        if str(CID4_order[i]) in dict:
            pro_temp_list = dict[str(CID4_order[i])]
            protein_list.extend(pro_temp_list)
            poly_temp_list = [i for x in pro_temp_list]
            poly_d_list.extend(poly_temp_list)

        if str(CID5_order[i]) in dict:
            pro_temp_list = dict[str(CID5_order[i])]
            protein_list.extend(pro_temp_list)
            poly_temp_list = [i for x in pro_temp_list]
            poly_d_list.extend(poly_temp_list)

    protein = torch.LongTensor(protein_list)
    poly_d_list = [i+19081 for i in poly_d_list]
    poly_d = torch.LongTensor(poly_d_list)
    edge_index = torch.stack((protein,poly_d),0)
    return edge_index

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# label

def create_5test_set_label():
    data = pd.read_csv(r'5_last.csv',encoding='utf-8')
    label = data['Odds Ratio']
    label = label.tolist()
    label = [0 if i < 2 else 1 for i in label]
    label = torch.LongTensor(label)
    return label

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
def main():
    data_dict = {}
    data_dict['ppi_edge_index'] = create_ppi_edge_index()
    # data_dict['p2d_edge_index'] = create_p2d_edge_index()
    # data_dict['ddi_pos_edge_index'] , data_dict['ddi_neg_edge_index'] = create_ddi_edge_index()
    # data_dict['ddi_pos_edge_weight'],data_dict['ddi_neg_edge_weight'] = create_ddi_edge_weight()

    # train_dict = {}
    # train_dict['train_d2poly_d_edge_index'] = create_train_d2poly_d_edge_index()
    # train_dict['train_p2poly_d_edge_index'] = create_train_protein2poly_d_edge_index()
    # train_dict['train_poly_ddi_edge_index'] = create_train_poly_ddi_edge_index()
    # train_dict['train_poly_ddi_edge_index_OR'] = create_train_poly_ddi_edge_index_OR()
    # train_dict['train_set_label'] =  create_train_set_label()

    # test3_dict = {}
    # test3_dict['test_d2poly_d_edge_index'] = create_test_d2poly_d_edge_index()
    # test3_dict['test_p2poly_d_edge_index'] = create_test_p2poly_d_edge_index()
    # test3_dict['test_set_label'] = create_test_set_label()

    # test4_dict = {}
    # test4_dict['test_d2poly_d_edge_index'] = create_4test_d2poly_d_edge_index()
    # test4_dict['test_p2poly_d_edge_index'] = create_4test_p2poly_d_edge_index()
    # test4_dict['test_set_label'] = create_4test_set_label()
    #
    # test5_dict = {}
    # test5_dict['test_d2poly_d_edge_index'] = create_5test_d2poly_d_edge_index()
    # test5_dict['test_p2poly_d_edge_index'] = create_5test_p2poly_d_edge_index()
    # test5_dict['test_set_label'] = create_5test_set_label()

    ######## 存储到pickle文件
    # import pickle
    # file = open('data_dict.pickle', 'wb')
    # pickle.dump(data_dict, file)
    # file.close()
    #
    # import pickle
    # with open('data_dict.pickle', 'rb') as file:
    #     dict = pickle.load(file)
    # print(dict)

    ######### 存储到pth文件
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.save(data_dict, "data_dict.pth")
    # # torch.save(test3_dict, "test3_dict.pth")
    # torch.save(test4_dict, "test4_dict.pth")
    # torch.save(test5_dict, "test5_dict.pth")
    # data_dict = torch.load("data_dict.pth", map_location=device)
    # print(data_dict)










if __name__ == "__main__":
    main()

















