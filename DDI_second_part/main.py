# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import time
import torch
from demo.model import *


def argument():
    parser = argparse.ArgumentParser()
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--optimizer', type=str, default='AdamW')  # 备选 'Adam' 'AdamW'  'SGD'
    parser.add_argument('--predict_num_layers', type=int, default=2)
    parser.add_argument('--predictor_name', type=str, default='MLPCAT')  # 备选  'MLP' 'MLPCAT'
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--layer1_num_layers', type=int, default=2)
    parser.add_argument('--layer1_num_nodes', type=int, default=19081)
    parser.add_argument('--layer1_encoder_name', type=str, default='TRANSFORMER') ## 备选'GCN' ,'WSAGE','TRANSFORMER','SAGE','GAT'
    parser.add_argument('--p_dim', type=int, default=128)  #######################
    parser.add_argument('--layer2_num_layers', type=int, default=2)
    parser.add_argument('--layer2_num_nodes', type=int, default=267)
    parser.add_argument('--layer2_encoder_name',  type=str, default='WSAGE') ## 备选'GCN' ,'WSAGE',
    parser.add_argument('--d_dim', type=int, default=55)   #######################
    parser.add_argument('--layer3_num_layers', type=int, default=2)
    parser.add_argument('--layer3_num_nodes', type=int, default=800)
    parser.add_argument('--poly_d_dim', type=int, default=32)  #######################
    parser.add_argument('--K_Fold', type=int, default=5)
    parser.add_argument('--K_Fold_order', type=int, default=0)
    parser.add_argument('--test_layer3_num_nodes', type=int, default=0)
    parser.add_argument('--loss_name', type=str, default='LSR')  ## 备选 'CE' 'LSCE' 'LSR'
    parser.add_argument('--use_ppi_neg', type=str2bool, default=False)
    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    args = argument()

    for run in range(args.K_Fold):

        fpr_list, tpr_list = [0],[0]
        print("cross validation: {} fold".format(args.K_Fold))
        print("***********K_Fold : {}****************".format(run))
        args.K_Fold_order = run
        model = BaseModel(args)
        optimizer = model.__prepare_optimizer__(args)
        model.param_init()
        model.__prepare_train_data__(args)
        start_time = time.time()
        for epoch in range(1, 1 + args.epochs):
            optimizer.zero_grad()
            loss,p_embedding,d_embedding = model.train(args)
            loss.backward()
            optimizer.step()

            if epoch%50 == 0:
                model.__prepare_train_data__(args)

            if epoch%100 == 0:
                print("*************epoch:{}*************".format(epoch))
                model.check()
                save_files = {
                    'p_embedding': p_embedding,
                    'd_embedding': d_embedding,
                    'd2poly_d_encoder': model.d2poly_d_encoder.state_dict(),
                    'p2poly_d_encoder': model.p2poly_d_encoder.state_dict(),
                    'linear': model.predictor.state_dict(),
                    'optimizer': optimizer.state_dict(), }
                torch.save(save_files, "./save_weights/K-Fold-{}-epoch:{}.pth".format(run, epoch))

                with torch.no_grad():
                    model_dict = torch.load("./save_weights/K-Fold-{}-epoch:{}.pth".format(run, epoch), map_location=args.device)
                    p_embedding = model_dict['p_embedding']
                    d_embedding = model_dict['d_embedding']
                    model_test = BaseModel(args)
                    optimizer_test = model_test.__prepare_optimizer__(args)
                    model_test.d2poly_d_encoder.load_state_dict(model_dict['d2poly_d_encoder'])
                    model_test.p2poly_d_encoder.load_state_dict(model_dict['p2poly_d_encoder'])
                    model_test.predictor.load_state_dict(model_dict['linear'])
                    optimizer_test.load_state_dict(model_dict['optimizer'])
                    model_test.__prepare_test_data__(args)
                    model_test.test(args, p_embedding, d_embedding)
                    fpr,tpr = model_test.check()
                    fpr_list.append(fpr)
                    tpr_list.append(tpr)
            if epoch == args.epochs:
                drow_acu_curve(fpr_list, tpr_list)
                plt.savefig("pictures/roc_K_Fold:{}.png".format(run), dpi=800)
        print('K-Fold:{}   共用时间 spent_time:{:.2f}min'.format(run,(time.time() - start_time)/60))



if __name__ == "__main__":
    main()




