from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
from time import time
import math
import random

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag










def sigmoid_function(z):
    fz=[]
    for num in z:
        fz.append(1/(1+math.exp(-num)))
    return fz




def test(model, user_dict, n_params,datapath,test_cf,testneg_cf):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.,
              'aupr':0.
              }

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']
    global path
    path=datapath
    #print(n_items,n_users)

    global train_user_set, test_user_set, testneg_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    testneg_user_set = user_dict['testneg_user_set']

    pool = multiprocessing.Pool(cores)
    test_pos_score=[]
    test_neg_score=[]
    test_pos_pair=[]
    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    testpos_users = list(test_user_set.keys())
    
    test_users=testpos_users
    
    #  列出所有test的user
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    #print(n_test_users)

    count = 0

    entity_gcn_emb, user_gcn_emb = model.generate()
    for u_id,i_id in test_cf:
        test_pos_score.append(model.rating(user_gcn_emb[int(u_id)], entity_gcn_emb[int(i_id)]).detach().cpu())
    for u_id,i_id in testneg_cf:
        test_neg_score.append(model.rating(user_gcn_emb[int(u_id)], entity_gcn_emb[int(i_id)]).detach().cpu())
    truthpos=[1 for x in range(0,len(test_pos_score))]
    truthneg=[0 for x in range(0,len(test_neg_score))]
    truthpos.extend(truthneg)
    
    test_pos_score.extend(test_neg_score)
    test_pos_score=sigmoid_function(test_pos_score)
    #print(truthpos,test_pos_score)
    aupr=AUPR(ground_truth=truthpos,prediction=test_pos_score)
    auc=AUC(ground_truth=truthpos, prediction=test_pos_score)
    #print(aupr,auc)

    
        
    pool.close()
    return auc,aupr



