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


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    #print(K_max_item_score)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    
    return r, auc,K_max_item_score

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def get_aupr(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    aupr = AUPR(ground_truth=r, prediction=posterior)
    return aupr

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    aupr=get_aupr(item_score,user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks,aupr):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc,'aupr':aupr}

def sigmoid_function(z):
    fz=[]
    for num in z:
        fz.append(1/(1+math.exp(-num)))
    return fz

def negative_sampling(user_item, trainset,testset):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in trainset[user]:
                    if user not in testset.keys():
                        break
                    elif neg_item not in testset[user]:
                        break
            neg_items.append(neg_item)
        return neg_items

def test_one_user(x):
    count=1
    # user u's ratings for user u
    rating = x[0]
    #print(rating)
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    try:
        testing_items = test_user_set[u]
    except Exception:
        testing_items = []
    # user u's items in the test set
    user_pos_test=[]
    user_neg_test=[]
    if u in test_user_set.keys():
        user_pos_test = test_user_set[u]
    if u in testneg_user_set.keys():
        user_neg_test=testneg_user_set[u]
    #print(user_pos_test)
    n_postest_items=len(user_pos_test)
    #neg_items=[]
    #for i in range(n_postest_items):
        #while True:
            #neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
            #if neg_item not in user_pos_test:
                #if neg_item not in training_items:
                    #if neg_item not in testing_items:
                        #break
        #user_neg_test.append(neg_item)
    n_negtest_items=len(user_neg_test)
    #print(len(neg_items))
    #print(n_test_items)
    #print(u)
    #assert n_negtest_items==n_postest_items
    all_items = set(range(0, n_items))
    positive=[1 for x in range(0,n_postest_items)]
    
    negative=[0 for x in range(0,n_negtest_items)]
    
    test_items = list(all_items - set(training_items))
    item_score = []
    for i in user_pos_test:
        item_score.append(rating[i-n_users])
    for j in user_neg_test:
        item_score.append(rating[j-n_users])
    item_score=sigmoid_function(item_score)
    positive.extend(negative)
    
    #print(positive)
    #print(item_score)
    
    aupr=AUPR(ground_truth=positive,prediction=item_score)
    auc=AUC(ground_truth=positive, prediction=item_score)
    
    #print(aupr)
    #print(auc)

    if args.test_flag == 'part':
        r, auc,kmax = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
        #if aupr>0.95:
            #if n_postest_items>5:
                #print(u,kmax)
    else:
        r, auc,aupr = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    aupr=AUPR(ground_truth=positive,prediction=item_score)
    auc=AUC(ground_truth=positive, prediction=item_score)
    return get_performance(user_pos_test, r, auc, Ks,aupr)


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
    #testneg_users =list(testneg_user_set.keys())
    #test_users=list(set(testpos_users+testneg_users))
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

    
        



     #print(user_gcn_emb[596]==entity_gcn_emb[596])
    #print(len(entity_gcn_emb))

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        #print(len(user_batch))
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            #print(n_item_batchs)
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)


                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                item_batch=item_batch+n_users
                #print(item_batch)
                i_g_embddings = entity_gcn_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()
                #print(i_rate_batch.shape)

                rate_batch[:, i_start: i_end] = i_rate_batch
                #print(rate_batch.shape)
                
                

                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = entity_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        #print(user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
            result['aupr']+=re['aupr']/n_test_users
            

    assert count == n_test_users
    pool.close()
    return auc,aupr



