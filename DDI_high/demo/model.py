# -*- coding: utf-8 -*-

from demo.neg_link_sample import *
from demo.layer import *
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch.nn
from torch.utils.data import DataLoader
from demo.traindata_sample import *
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,average_precision_score
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc

class BaseModel(object):

    def __init__(self,args):

        self.__prepare_data__(args)

        self.train_data,self.test3_data = K_Fold(args)

        # Triple Layers
        self.bottom_encoder = create_gnn_layer(
            input_channels=args.p_dim,
            hidden_channels=args.p_dim,
            num_layers=args.layer1_num_layers,
            dropout=args.dropout,
            encoder_name=args.layer1_encoder_name).to(args.device)

        self.p2d_encoder = p2d_HierarchyConv(args).to(args.device)

        self.middle_encoder = create_gnn_layer(
            input_channels=2*args.d_dim,
            hidden_channels=args.d_dim,
            num_layers=args.layer2_num_layers,
            dropout=args.dropout,
            encoder_name=args.layer2_encoder_name).to(args.device)

        self.d2poly_d_encoder = d2poly_d_HierarchyConv(args).to(args.device)

        self.p2poly_d_encoder = p2poly_d_HierarchyConv(args).to(args.device)

        self.predictor = create_predictor_layer(
            hidden_channels=args.poly_d_dim,
            num_layers=args.predict_num_layers, # 2
            dropout=args.dropout, # 0.0
            predictor_name=args.predictor_name).to(args.device)

        if args.loss_name == 'CE':
            self.loss = torch.nn.CrossEntropyLoss()
        elif args.loss_name == 'LSCE':
            self.loss = LabelSmoothingCrossEntropy()
        elif args.loss_name == 'LSR':
            self.loss = LabelSmoothingRegularization()
        else:
            raise ValueError("loss computer Wrong Setting!")

    def __prepare_optimizer__(self,args):
        # Parameters and Optimizer
        self.para_list = list(self.bottom_encoder.parameters()) + list(self.p2d_encoder.parameters()) + \
                         list(self.middle_encoder.parameters()) + list(self.d2poly_d_encoder.parameters()) + \
                         list(self.p2poly_d_encoder.parameters()) + list(self.predictor.parameters())


        if args.optimizer == 'AdamW':   # # Adam
            self.optimizer = torch.optim.AdamW(self.para_list, lr=args.lr)
        elif args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.para_list, lr=args.lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        elif args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.para_list, lr=args.lr)
        else:
            raise ValueError('optimizer Wrong Setting!')
        return self.optimizer

    def __prepare_data__(self,args):
        data_dict = torch.load("dataset/data_dict.pth", map_location=args.device)
        self.ppi_edge_index = data_dict['ppi_edge_index']
        if args.use_ppi_neg == True:
            self.ppi_neg_edge_index = neg_sample(self.ppi_edge_index, args.layer1_num_nodes,
                                                        3*self.ppi_edge_index.shape[1], method='sparse')
        self.p2d_edge_index = data_dict['p2d_edge_index']
        self.ddi_pos_edge_index = data_dict['ddi_pos_edge_index']
        self.ddi_neg_edge_index = data_dict['ddi_neg_edge_index']
        self.ddi_pos_edge_weight = data_dict['ddi_pos_edge_weight']
        self.ddi_neg_edge_weight = data_dict['ddi_neg_edge_weight']
        init_d_dict = torch.load("dataset/init_d_dict.pth", map_location=args.device)
        self.init_d_embedding = init_d_dict['init_d_embedding']

    def __prepare_train_data__(self,args):

        self.train_d2poly_d_edge_index,self.train_p2poly_d_edge_index,\
        self.train_smooth_label,self.label =train_sample(args,self.train_data)

    def __prepare_test3_data__(self, args):

        self.test_d2poly_d_edge_index = create_d2poly_d_edge_index(self.test3_data,args)
        self.test_p2poly_d_edge_index = create_p2poly_d_edge_index(self.test3_data,args)
        self.label = create_label(self.test3_data,args)
        self.test_split_num = 0
        for i in range(self.test3_data.shape[0]):
            if self.test3_data['Odds Ratio'][i] > 2:
                self.test_split_num = i
                break
        assert self.test_split_num != 0


    def __prepare_test4_data__(self, args):

        test4_dict = torch.load("dataset/test4_dict.pth", map_location=args.device)
        self.test_d2poly_d_edge_index = test4_dict['test_d2poly_d_edge_index']
        self.test_p2poly_d_edge_index = test4_dict['test_p2poly_d_edge_index']
        self.label = test4_dict['test_set_label']

    def __prepare_test5_data__(self, args):

        test4_dict = torch.load("dataset/test5_dict.pth", map_location=args.device)
        self.test_d2poly_d_edge_index = test4_dict['test_d2poly_d_edge_index']
        self.test_p2poly_d_edge_index = test4_dict['test_p2poly_d_edge_index']
        self.label = test4_dict['test_set_label']

    def param_init(self):
        self.bottom_encoder.reset_parameters()
        self.p2d_encoder.reset_parameters()
        self.middle_encoder.reset_parameters()
        self.d2poly_d_encoder.reset_parameters()

    def check(self):

        label = self.label.detach().cpu().numpy().tolist()
        predict = self.predict.detach().cpu().numpy().tolist()

        cm = confusion_matrix(label, predict)
        import seaborn as sns
        sns.heatmap(cm,cmap="YlGnBu_r",fmt="d",annot=True)

        # acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        # pre = cm[0][0] / (cm[0][0] + cm[1][0])
        # rec = cm[0][0] / (cm[0][0] + cm[0][1])
        # auc = roc_auc_score(label, predict)
        # F1 = 2 * pre * rec / (pre + rec)
        # print("acc:{:.3f}".format(acc),"pre:{:.3f}".format(pre),"rec:{:.3f}".format(rec),
        #       "AUROC:{:.3f}".format(auc),"F1:{:.3f}".format(F1),cm[0][0],cm[0][1], cm[1][0],cm[1][1])

        # print(cm[1][1], cm[1][0])
        # print(cm[0][1], cm[0][0])


        acc = accuracy_score(label, predict)
        auprc = average_precision_score(label, predict)
        pre = precision_score(label, predict)
        rec = recall_score(label, predict)
        F1 = f1_score(label, predict)
        auroc = roc_auc_score(label, predict)
        print("acc:{:.3f}".format(acc),"pre:{:.3f}".format(pre),"rec:{:.3f}".format(rec),
              "AUROC:{:.3f}".format(auroc),"AUPRC:{:.3f}".format(auprc),"F1:{:.3f}".format(F1),)
        fpr, tpr, threshold = roc_curve(label, predict)

        # J = tpr-fpr
        # idx = np.argmax(J)
        # best_threshold = threshold[idx]
        # print("best_threshold",best_threshold)

        return fpr[1],tpr[1]

    def train(self, args):

        # self.__prepare_train_data__(args)
        embed = torch.nn.Embedding(args.layer1_num_nodes, args.p_dim)
        self.p_embedding = embed(torch.LongTensor([i for i in range(args.layer1_num_nodes)])).to(args.device)

        # ppi
        bottom_graph = Data(x=self.p_embedding, edge_index=self.ppi_edge_index, edge_attr=None, num_nodes=args.layer1_num_nodes)
        bottom_loader = NeighborLoader(bottom_graph, batch_size=1024, num_neighbors=[25, 10])
        for _, batch in enumerate(bottom_loader):
            self.p_embedding = self.bottom_encoder(self.p_embedding,batch.edge_index)

        if args.use_ppi_neg == True:
            bottom_graph = Data(x=self.p_embedding, edge_index=self.ppi_neg_edge_index, edge_attr=None, num_nodes=args.layer1_num_nodes)
            bottom_loader = NeighborLoader(bottom_graph, batch_size=1024, num_neighbors=[25, 10])
            for _, batch in enumerate(bottom_loader):
                self.p_embedding2 = self.bottom_encoder(self.p_embedding,batch.edge_index)
            self.p_embedding = self.p_embedding - self.p_embedding2

        ## p2d
        d_init_matrix = torch.zeros(args.layer2_num_nodes, args.p_dim).to(args.device)
        self.pd_embedding = torch.cat((self.p_embedding, d_init_matrix), 0)
        self.d_embedding = self.p2d_encoder(self.pd_embedding,self.p2d_edge_index)
        self.d_embedding = torch.cat((self.d_embedding,self.init_d_embedding),1)

        ## ddi
        pos_embedding = self.middle_encoder(self.d_embedding,self.ddi_pos_edge_index, self.ddi_pos_edge_weight)
        neg_embedding = self.middle_encoder(self.d_embedding,self.ddi_neg_edge_index, self.ddi_neg_edge_weight)
        self.d_embedding = pos_embedding - neg_embedding


        ## d2poly_d
        poly_d_init_matrix = torch.zeros(args.layer3_num_nodes, args.d_dim).to(args.device)
        self.dpd_embedding = torch.cat((self.d_embedding, poly_d_init_matrix), 0)
        self.poly_d_embedding1 = self.d2poly_d_encoder(self.dpd_embedding, self.train_d2poly_d_edge_index)

        ## p2poly_d
        poly_d_init_matrix = torch.zeros(args.layer3_num_nodes,args.p_dim).to(args.device)
        self.ppd_embedding = torch.cat((self.p_embedding, poly_d_init_matrix), 0)
        self.poly_d_embedding2 = self.p2poly_d_encoder(self.ppd_embedding, self.train_p2poly_d_edge_index)

        ## predict
        self.poly_d_embedding = self.predictor(self.poly_d_embedding1, self.poly_d_embedding2)

        ## compute loss
        loss = self.loss(self.poly_d_embedding,self.label)

        self.predict = self.poly_d_embedding.argmax(-1)

        return loss,self.p_embedding,self.d_embedding


    @torch.no_grad()
    def test(self, args,p_embedding,d_embedding):

        ## d2poly_d
        poly_d_init_matrix = 0
        if args.test_order == 3:
            poly_d_init_matrix = torch.zeros((args.test3_layer3_num_nodes ,args.d_dim)).to(args.device)
        elif args.test_order == 4:
            poly_d_init_matrix = torch.zeros((args.test4_layer3_num_nodes , args.d_dim)).to(args.device)
        elif args.test_order == 5:
            poly_d_init_matrix = torch.zeros((args.test5_layer3_num_nodes, args.d_dim)).to(args.device)
        self.dpd_embedding = torch.cat((d_embedding,poly_d_init_matrix),0)
        self.poly_d_embedding1 = self.d2poly_d_encoder(self.dpd_embedding, self.test_d2poly_d_edge_index)

        ## p2poly_d
        poly_d_init_matrix = 0
        if args.test_order == 3:
            poly_d_init_matrix = torch.zeros((args.test3_layer3_num_nodes ,args.p_dim)).to(args.device)
        elif args.test_order == 4:
            poly_d_init_matrix = torch.zeros((args.test4_layer3_num_nodes, args.p_dim)).to(args.device)
        elif args.test_order == 5:
            poly_d_init_matrix = torch.zeros((args.test5_layer3_num_nodes, args.p_dim)).to(args.device)
        self.ppd_embedding = torch.cat((p_embedding,poly_d_init_matrix),0)
        self.poly_d_embedding2 = self.p2poly_d_encoder(self.ppd_embedding, self.test_p2poly_d_edge_index)

        ## predict
        self.poly_d_embedding = self.predictor(self.poly_d_embedding1, self.poly_d_embedding2)

        self.predict = self.poly_d_embedding.argmax(-1)




def K_Fold(args):


    K = args.K_Fold
    n = args.K_Fold_order
    data = pd.read_csv("dataset/3_order.csv")
    nums = data.shape[0] - data.shape[0] % K
    data = data[:nums]

    all_list = [i for i in range(0,nums)]
    basic_list = [i for i in range(0,nums,K)]
    test_list = [i+n for i in basic_list]
    train_list = [i for i in all_list if i not in test_list]

    train_data = data.iloc[train_list]
    train_data.index = range(len(train_list))
    test_data = data.iloc[test_list]
    test_data.index = range(len(test_list))

    args.test3_layer3_num_nodes = test_data.shape[0]
    return train_data,test_data

def create_gnn_layer(input_channels, hidden_channels, num_layers, dropout=0, encoder_name='SAGE'):
    encoder_name = encoder_name.upper()
    if encoder_name.upper() == 'GCN':
        return GCN(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'WSAGE':
        return WSAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'TRANSFORMER':
        return Transformer(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'SAGE':
        return SAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'GAT':
        return GAT(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    else:
        raise ValueError('GNN layer Wrong Setting!')




def create_predictor_layer(hidden_channels, num_layers, dropout=0, predictor_name='MLP'):
    predictor_name = predictor_name.upper()
    if predictor_name == 'MLP':
        return MLPPredictor(hidden_channels, hidden_channels, 2, num_layers, dropout)
    elif predictor_name == 'MLPCAT':
        return MLPCatPredictor(hidden_channels, hidden_channels, 2, num_layers, dropout)
    else:
        raise ValueError('predictor layer Wrong Setting!')


def drow_acu_curve(fpr_list, tpr_list):

    lw = 1
    plt.figure(figsize=(10, 10))
    plt.plot(fpr_list, tpr_list, color='darkorange',lw=lw,label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
