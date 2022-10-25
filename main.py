'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import copy
import torch
import numpy as np
import math

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.KGIN import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
import collections
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from deepctr.models import NFM,IFM,DIFM,AFM
from deepctr.feature_column import SparseFeat,DenseFeat,get_feature_names
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam,Adagrad,Adamax
from sklearn.decomposition import PCA
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0



#descriptors preparation


def get_feed_dict(train_entity_pairs, start, end, train_user_set,trainneg_user_set):

    def negative_sampling1(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                #print(n_users,n_items)
                neg_item = np.random.randint(low=n_users, high=n_items+n_users, size=1)[0]
                if neg_item not in train_user_set[user]:
                    if user not in test_user_set.keys():
                            break
                    else:
                        if neg_item not in test_user_set[user]:
                            break
            neg_items.append(neg_item)
        return neg_items
    
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                #print(n_users,n_items)
                neg_item = np.random.randint(low=n_users, high=n_items+n_users, size=1)[0]
                if neg_item not in train_user_set[user]:
                     break
            neg_items.append(neg_item)
        return neg_items
    
    def get_negative(user_item, train_user_set,trainneg_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                #print(trainneg_user_set[user])
                if user not in trainneg_user_set.keys():
                    neg_item = np.random.randint(low=n_users, high=n_items+n_users, size=1)[0]
                    if neg_item not in train_user_set[user]:
                        break
                else:
                    neg_item = np.random.choice(trainneg_user_set[user])
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(get_negative(entity_pairs,train_user_set,trainneg_user_set)).to(device)
    #feed_dict['neg_items'] = torch.LongTensor(negative_sampling1(entity_pairs,train_user_set)).to(device)
    #print(feed_dict['pos_items'],feed_dict['neg_items'])
    return feed_dict


def KGIN(i):
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device,n_items,n_users
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    print(device)
    
    """build dataset"""
    datapath,train_cf, test_cf,trainneg_cf, testneg_cf, user_dict, n_params, graph, mat_list = load_data(args,i)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list
    
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    
    print(n_users,n_items,n_entities,n_relations,n_nodes)
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
    
    """define model"""
    mean_mat_list_user = mean_mat_list[0].tocsr()[:n_users, :].tocoo()
    mean_mat_list_item = mean_mat_list[0].tocsr()[n_users:n_items+n_users, :].tocoo()
    model = Recommender(n_params, args, graph, mean_mat_list_user,mean_mat_list_item).to(device)
    
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_model=None
    cur_best_pre_0 = 0
    cur_best_pre_1=0
    stopping_step = 0
    should_stop = False
    
    print("start knowledge graph training ...")
    
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]
        
        #useddata=[]
        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs,
                            s, s + args.batch_size,
                            user_dict['train_user_set'],user_dict['trainneg_user_set'])
            
            #print(len(batch['pos_items']))
            #print(len(batch['neg_items']))
            #print(len(list(user_dict['test_user_set'].keys())))
            batch_loss, _, _, batch_cor = model(batch)

            batch_loss = batch_loss
            #print(batch_loss)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            cor_loss += batch_cor
            s += args.batch_size

        train_e_t = time()
        
        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            test_s_t = time()
            res_auc,res_aupr = test(model, user_dict, n_params,datapath,test_cf,testneg_cf)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "AUC","AUPR"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), res_auc,res_aupr]
            )
            print(train_res)
            
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop,cur_best_pre_1 = early_stopping(res_aupr, cur_best_pre_0,
                                                                    stopping_step,res_auc,cur_best_pre_1, expected_order='acc',
                                                                    flag_step=10)
            #if stopping_step==0:
                #best_model=copy.deepcopy(model)
            
            if should_stop:
                break

            """save weight"""
            if res_aupr == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_e_t - train_s_t, epoch, loss.item(), cor_loss.item()))
    

   
    print('early stopping at %d, AUPR:%.4f, AUC:%.4f' % (epoch, cur_best_pre_0,cur_best_pre_1))
    return(cur_best_pre_0,cur_best_pre_1,test_cf,testneg_cf)

def sigmoid_function(z):
    fz=[]
    for num in z:
        fz.append(1/(1+math.exp(-num)))
    return fz

if __name__=='__main__':
    AUPR=[]
    AUC=[]
    for i in range(1,11):
        print(i)
        AUPR1,AUC1,test_cf,testneg_cf=KGIN(i)
        AUPR.append(AUPR1)
        AUC.append(AUC1)
    #test_pos_score=[]
    #test_neg_score=[]
    #savemodel,AUPR1,AUC1,test_cf,testneg_cf=KGIN(i)
    #entity_gcn_emb,user_gcn_emb=savemodel.generate()
    #for u_id,i_id in test_cf:
        #test_pos_score.append(savemodel.rating(user_gcn_emb[int(u_id)], entity_gcn_emb[int(i_id)]).detach().cpu())
    #for u_id,i_id in testneg_cf:
        #test_neg_score.append(savemodel.rating(user_gcn_emb[int(u_id)], entity_gcn_emb[int(i_id)]).detach().cpu())
    
    #test_pos_score.extend(test_neg_score)
    #test_pos_score=sigmoid_function(test_pos_score)
    #fw1=open('data/Y08/warm_start_1_10/test_result_'+str(i)+'.txt','w')
    #fw1=open('data/luo/protein_coldstart/test_result_'+str(i)+'.txt','w')
    #for score in test_pos_score:
        #fw1.write(str(score))
        #fw1.write('\n')
    #print(truthpos,test_pos_score)
    #AUPR.append(AUPR1)
    #AUC.append(AUC1)
    print(AUPR,AUC)

    
      