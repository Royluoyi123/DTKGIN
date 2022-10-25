import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
trainneg_user_set = defaultdict(list)
testneg_user_set = defaultdict(list)


def read_cf(file_name):
    posinter_mat = list()
    neginter_mat = list()
    lines = open(file_name, "r").readlines()
    for line in open(file_name,'r'):
        lines=line.split()
        if int(lines[3])==1:
            posinter_mat.append([lines[0], lines[2]])
        else:
            neginter_mat.append([lines[0], lines[2]])
    return np.array(posinter_mat),np.array(neginter_mat)

def read_cf1(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)

def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = 709
    n_items = 1512
    #n_users = 791
    #n_items = 989

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))

def remap_item2(train_data, test_data):

    for u_id, i_id in train_data:
        trainneg_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        testneg_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)
    #print(len(can_triplets_np))

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()
    #print(triplets)
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities=12015
    #n_nodes = n_entities=25487
    n_relations = max(triplets[:, 1]) + 1
    #print(n_relations)
    #print(n_entities)
    #assert n_entities==95160
    #assert n_relations==48

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            #print(cf)
            #cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            #print(cf)
            vals = [1.] * len(cf)
            #print(vals)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    #print(norm_mat_list[0])
    #norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, :].tocoo()
    #print(norm_mat_list[0])
    #print(norm_mat_list[0].shape)
    #mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, :].tocoo()
    #print(mean_mat_list[0].nnz)

    return adj_mat_list, norm_mat_list, mean_mat_list

    

def load_data(model_args,i):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf,trainneg_cf = read_cf(directory + 'train_fold_'+str(i)+'.txt')
    test_cf,testneg_cf = read_cf(directory + 'test_fold_'+str(i)+'.txt')
    #print(train_cf)
    #train_cf = read_cf1(directory + 'trainpos1.txt')
    #test_cf = read_cf1(directory + 'testpos1.txt')
    #trainneg_cf = read_cf1(directory + 'trainneg1.txt')
    #testneg_cf = read_cf1(directory + 'testneg1.txt')
    print(len(train_cf),len(trainneg_cf),len(test_cf),len(testneg_cf))
    
    remap_item(train_cf, test_cf)
    remap_item2(trainneg_cf, testneg_cf)
    #print(len(testneg_user_set.keys()))
    print('combinating train_cf and kg data ...')
    triplets = read_triplets("data/luo/kg_final.txt")
    #triplets = read_triplets("data/luodataset/kg_final.txt")
    #print(len(triplets))
    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)
    
    print(n_users,n_items,n_nodes,n_entities,n_relations)
    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
        'trainneg_user_set': trainneg_user_set,
        'testneg_user_set': testneg_user_set,
    }
    print(len(test_user_set.keys()))
    print(len(testneg_user_set.keys()))
    print(len(train_user_set.keys()))
    print(len(trainneg_user_set.keys()))
    a=list(train_user_set.keys())
    a.sort()
    b=list(trainneg_user_set.keys())
    b.sort()
    print(a==b)
    a=list(test_user_set.keys())
    a.sort()
    b=list(testneg_user_set.keys())
    b.sort()
    print(a==b)
    #print(len(train_user_set.keys()),len(test_user_set.keys()),len(trainneg_user_set.keys()),len(testneg_user_set.keys()))
    return directory,train_cf,test_cf,trainneg_cf,testneg_cf, user_dict, n_params, graph, \
           [adj_mat_list, norm_mat_list, mean_mat_list]

