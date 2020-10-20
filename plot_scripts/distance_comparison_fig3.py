import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
import json
import random
import torch
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from collections import defaultdict
from plot_scripts.try_networkx import preprocess_adj_op, gen_graph, edge_match, node_match
import networkx as nx

from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C
from ConfigSpace.util import get_one_exchange_neighbourhood
from nasbench.lib import graph_util
from nasbench import api
from preprocessing.gen_json import transform_operations


from models.configs import configs
from utils.utils import load_json, preprocessing
from models.model import Model

import pickle
import seaborn

"""Autocorrelation plot for architecture encoding."""


def config2data_B(config, b):
    VERTICES = 7
    MAX_EDGES = 9
    budget = 108

    bitlist = [0] * (VERTICES * (VERTICES - 1) // 2)
    for i in range(MAX_EDGES):
        bitlist[config["edge_%d" % i]] = 1
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit

    matrix = np.fromfunction(graph_util.gen_is_edge_fn(out),
                             (VERTICES, VERTICES),
                             dtype=np.int8)
    # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
    if graph_util.num_edges(matrix) > MAX_EDGES:
        return None, None

    labeling = [config["op_node_%d" % i] for i in range(5)]
    labeling = ['input'] + list(labeling) + ['output']
    model_spec = api.ModelSpec(matrix, labeling)
    try:
        data = b.dataset.query(model_spec, epochs=budget)
        msp = b.dataset.get_metrics_from_spec(model_spec)
    except api.OutOfDomainError:
        return None, None
    test_acc = [msp[1][108][k]['final_test_accuracy'] for k in range(3)]
    return data, test_acc


def config2data_A(config, b):
    VERTICES = 7
    MAX_EDGES = 9
    budget = 108

    matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
    idx = np.triu_indices(matrix.shape[0], k=1)
    for i in range(VERTICES * (VERTICES - 1) // 2):
        row = idx[0][i]
        col = idx[1][i]
        matrix[row, col] = config["edge_%d" % i]

    # if not graph_util.is_full_dag(matrix) or graph_util.num_edges(matrix) > MAX_EDGES:
    if graph_util.num_edges(matrix) > MAX_EDGES:
        return None, None

    labeling = [config["op_node_%d" % i] for i in range(5)]
    labeling = ['input'] + list(labeling) + ['output']
    model_spec = api.ModelSpec(matrix, labeling)
    try:
        data = b.dataset.query(model_spec, epochs=budget)
        msp = b.dataset.get_metrics_from_spec(model_spec)
    except api.OutOfDomainError:
        return None, None

    test_acc = [msp[1][108][k]['final_test_accuracy'] for k in range(3)]

    return data, np.mean(test_acc)

def data2mat(data):
    adj = data['module_adjacency'].tolist()
    op = data['module_operations']
    if len(adj) <= 6:
        for row in range(len(adj)):
            for _ in range(7 - len(adj)):
                adj[row].append(0)
        for _ in range(7 - len(adj)):
            adj.append([0, 0, 0, 0, 0, 0, 0])
    op = transform_operations(op)
    return np.array(adj).astype(float), op.astype(float)


def load_json(f_name):
    with open(f_name) as f:
        data = json.load(f)
    return data

def edit_distance(arch_1, arch_2):
    adj_1, ops_1 = arch_1[0], arch_1[1]
    adj_2, ops_2 = arch_2[0], arch_2[1]
    adj_dist = np.sum(np.array(adj_1) != np.array(adj_2))
    ops_dist = np.sum(np.array(ops_1) != np.array(ops_2))
    return int(adj_dist + ops_dist)

def edit_distance_yu(arch_1, arch_2):
    adj_1, ops_1 = arch_1[0], arch_1[1]
    adj_2, ops_2 = arch_2[0], arch_2[1]
    adj_1, ops_1 = preprocess_adj_op(adj_1, ops_1)
    adj_2, ops_2 = preprocess_adj_op(adj_2, ops_2)
    G1 = gen_graph(adj_1, ops_1)
    G2 = gen_graph(adj_2, ops_2)
    return int(nx.graph_edit_distance(G1, G2, node_match=node_match, edge_match=edge_match))


def l2_distance(feat_1, feat_2):
    return np.linalg.norm(feat_1-feat_2, ord=2)


def mutate(config, b, N_node=None):
    if isinstance(N_node, random_nodes):
        N_node = N_node.random()
        print('Node={}, type={}'.format(N_node, N_node.__class__))
    is_valid_graph = False
    satisfy_num_node_constraint = False
    while (not is_valid_graph) or (not satisfy_num_node_constraint):
        neighbor_gen = get_one_exchange_neighbourhood(config, seed=random.randint(1, 1e6))
        neighbors = list(neighbor_gen)
        neighbor_config = neighbors[random.randint(0, len(neighbors)-1)]
        data, _ = config2data_A(neighbor_config,b)

        # Determine is valid graph
        if data is None:
            is_valid_graph = False
            print('Invalid graph')
        else:
            is_valid_graph = True
            # Determine if the graph satisfy number of nodes constraint
            num_node = len(data['module_operations'])
            if N_node is None:
                satisfy_num_node_constraint = True
            elif isinstance(N_node, list):
                satisfy_num_node_constraint = True if num_node in N_node else False
            elif isinstance(N_node, int):
                satisfy_num_node_constraint = True if num_node==N_node else False
            else:
                raise ValueError('Unrecognized N_node')
            print('sampled {}'.format(num_node))
    print('Architecture length is {}'.format(num_node))
    return neighbor_config

def config2embedding(config, b):
    data, _ = config2data_A(config, b)
    adj, op = data2mat(data)
    adj_, op_ = adj, op
    adj, op = torch.Tensor(adj).unsqueeze(0).cuda(), torch.Tensor(op).unsqueeze(0).cuda()
    adj, op, prep_reverse = preprocessing(adj, op, **cfg['prep'])
    embedding = model._encoder(op, adj)[0]
    return embedding, adj_, op_

class random_nodes(object):
    def __init__(self, nodes, p):
        assert len(nodes)==len(p), 'len(nodes) should be equal to len(p)'
        assert sum(p)==1, 'sum(p) should be 1'
        self.nodes = nodes
        self.p = p
    def random(self):
        return int(np.random.choice(self.nodes, p=self.p))

def random_walk(b, use_true_edit_distance = True):
    # initalize
    #random.seed(s)
    bin_size = 0.5

    cs = b.get_configuration_space()

    init_N_node = 7
    N_node = 7
    satisfy_num_node_constraint = False
    while not satisfy_num_node_constraint:
        config = cs.sample_configuration()
        data, _ = config2data_A(config, b)
        if data is None:
            satisfy_num_node_constraint = False
        else:
            num_node = len(data['module_operations'])
            if isinstance(init_N_node, list):
                satisfy_num_node_constraint = True if num_node in init_N_node else False
            elif isinstance(init_N_node, int):
                satisfy_num_node_constraint = True if num_node == init_N_node else False
    print('Successfully generated a valid initial graph!')

    embedding_list = []
    matrix_encoding_list = []
    test_accuracy_list = []
    random_walk_steps = 1000
    for count in range(random_walk_steps):
        print(count)
        config = mutate(config, b,  N_node=N_node)
        embedding, adj, op = config2embedding(config, b)
        embedding_list.append(embedding)
        matrix_encoding_list.append((adj, op))
        _, test_accuracy = config2data_A(config, b)
        test_accuracy_list.append(test_accuracy)

    MAX_EDIT_DISTANCE = 8
    EditDistance2L2Distance = defaultdict(list)
    EditDistance2L2Distance[0] = [0.0]
    EditDistance2accPair = defaultdict(list)
    RWA2accPair = defaultdict(list)
    L2Distance2accPair = defaultdict(list)
    L2Distance_accPair = []
    for k in range(1, MAX_EDIT_DISTANCE+1):
        for p1 in range(0, random_walk_steps-k):
            p2 = p1 + k
            L2 = l2_distance(embedding_list[p1].cpu().detach().numpy().squeeze().mean(axis=0),
                             embedding_list[p2].cpu().detach().numpy().squeeze().mean(axis=0))
            L2Distance2accPair[(L2-0.000001)//bin_size+1].append( (test_accuracy_list[p1], test_accuracy_list[p2]) )
            L2Distance_accPair.append((L2, test_accuracy_list[p1], test_accuracy_list[p2]))

            if use_true_edit_distance:
                edit_dist = edit_distance_yu(matrix_encoding_list[p1], matrix_encoding_list[p2])
                EditDistance2L2Distance[edit_dist].append(L2)
                print('{}/{}, Estimated Edit Distance = {}, True Edit Distance = {}'.format(p1, random_walk_steps, k, edit_dist))
                EditDistance2accPair[edit_dist].append( (test_accuracy_list[p1], test_accuracy_list[p2]) )
                RWA2accPair[k].append( (test_accuracy_list[p1], test_accuracy_list[p2]) )
            else:
                EditDistance2L2Distance[k].append(L2)
                EditDistance2accPair[k].append((test_accuracy_list[p1], test_accuracy_list[p2]))
                RWA2accPair[k].append((test_accuracy_list[p1], test_accuracy_list[p2]))
                print('Estimated Edit Distance = {}'.format(k))

    All_Data = {'EditDistance2L2Distance': EditDistance2L2Distance, 'L2Distance2accPair': L2Distance2accPair,
                'EditDistance2accPair':EditDistance2accPair, 'RWA2accPair':RWA2accPair, 'L2Distance_accPair':L2Distance_accPair,
                'bin_size':bin_size}

    pickle.dump(All_Data, open("data/RWA_nasbench101_2.pt", "wb"))
    draw_edit_distance_paper(EditDistance2L2Distance, L2Distance_accPair, EditDistance2accPair)

    return EditDistance2L2Distance


def plot_RWA(L2Distance2accPair, EditDistance2accPair, RWA2accPair, L2Distance_accPair, bin_size):

    # L2 Distance correlation
    indices = list(L2Distance2accPair.keys())
    indices.sort()
    cor = [(k*bin_size, cal_pearson_correlation(L2Distance2accPair[k])) for k in indices if len(L2Distance2accPair[k])>9 ]
    cor = np.array(cor)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cor[:,0], cor[:,1],'-o')
    ax.set_xlabel('L2 distance')
    ax.set_ylabel('Correlation')

    indices = list(EditDistance2accPair.keys())
    indices.sort()
    cor = [(k, cal_pearson_correlation(EditDistance2accPair[k])) for k in indices if len(EditDistance2accPair[k])>9 ]
    cor = np.array(cor)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cor[:,0], cor[:,1],'-o')
    ax.set_xlabel('Edit distance')
    ax.set_ylabel('Correlation')

    indices = list(RWA2accPair.keys())
    indices.sort()
    cor = [(k, cal_pearson_correlation(RWA2accPair[k])) for k in indices if len(RWA2accPair[k])>9 ]
    cor = np.array(cor)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cor[:, 0], cor[:, 1],'-o')
    ax.set_xlabel('RWA distance')
    ax.set_ylabel('Correlation')

    # plot the windowed l2 distance
    bs = 2.0
    L2Distance_accPair.sort(key = lambda x: x[0])
    cor = []
    for win_c in np.linspace(-bs/2.0+0.001, 6.0, num=100):
        win_low = max(0, win_c - bs/2.0)
        win_high = win_c + bs/2.0
        idx_low = find_edge(L2Distance_accPair, win_low)
        idx_high = find_edge(L2Distance_accPair, win_high)
        if idx_low == None or idx_high == None:
            break
        pair = [L2Distance_accPair[k][1:] for k in range(idx_low, idx_high+1)]
        cor.append( (win_c, cal_pearson_correlation(pair) ) )
    cor = np.array(cor)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cor[:, 0]+bs/2.0, cor[:, 1], '-')
    ax.set_xlabel('2 Distance, sliding window')
    ax.set_ylabel('Correlation')



def find_edge(l, p):
    for k in range(len(l)-1):
        if p == 0:
            return 0
        if l[k][0] <= p and p < l[k+1][0]:
            return k
    return None

def cal_pearson_correlation(pair):
    pair = np.array(pair)
    return np.corrcoef(pair, rowvar=False)[0,1]



def distance_compare_plot(EditDistance2L2Distance):
    EditDists = []
    L2Dists = []
    for edit_dist, l2_dist in EditDistance2L2Distance.items():
        EditDists.extend([edit_dist]*len(l2_dist))
        L2Dists.extend(l2_dist)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(EditDists, L2Dists)
    ax.set_xlabel('Edit distance')
    ax.set_ylabel('L2 distance')

    E = []
    L2 = []
    for edit_dist, l2_dist in EditDistance2L2Distance.items():
        print('Edit distance {}, has {} samples.'.format(edit_dist, len(l2_dist)))
        E.append(edit_dist)
        L2.append(l2_dist)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.violinplot(L2, E)
    ax.set_xlabel('Edit distance')
    ax.set_ylabel('L2 distance')

    D = [EditDistance2L2Distance[k] for k in range(9)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(D)
    ax.set_xlabel('Edit distance')
    ax.set_ylabel('L2 distance')

    STD = np.array([np.array(EditDistance2L2Distance[k]).std() for k in range(6+1)])
    MEAN = np.array([np.array(EditDistance2L2Distance[k]).mean() for k in range(6 + 1)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(7), MEAN)
    ax.plot(np.arange(7), MEAN+STD, '--')
    ax.plot(np.arange(7), MEAN - STD, '--')
    ax.set_xlabel('Edit distance')
    ax.set_ylabel('L2 distance')


def draw_edit_distance_paper(EditDistance2L2Distance, L2Distance_accPair, EditDistance2accPair):
    import matplotlib.pyplot as plt
    import seaborn
    import pickle
    import numpy as np
    with open('data/RWA_nasbench101_2.pt', 'rb') as f:
        All_Data = pickle.load(f)
    EditDistance2L2Distance = All_Data['EditDistance2L2Distance']
    keep_prob=0.1 # for randomly drop some outlier
    D = []
    D.append(EditDistance2L2Distance[0])
    D.append([EditDistance2L2Distance[1][k] for k in range(len(EditDistance2L2Distance[1]))])
    D.append([EditDistance2L2Distance[2][k] for k in range(len(EditDistance2L2Distance[2]))])
    D.append([EditDistance2L2Distance[3][k] for k in range(len(EditDistance2L2Distance[3]))])
    D.append([EditDistance2L2Distance[4][k] for k in range(len(EditDistance2L2Distance[4]))])
    D.append([EditDistance2L2Distance[5][k] for k in range(len(EditDistance2L2Distance[5]))])
    D.append([EditDistance2L2Distance[6][k] for k in range(len(EditDistance2L2Distance[6]))])
    D.append([EditDistance2L2Distance[7][k] for k in range(len(EditDistance2L2Distance[7]))])

    fig = plt.figure()
    ax = seaborn.boxplot(data=D, showfliers=True)
    ax.set_xlabel('Edit distance')
    ax.set_ylabel('L2 distance')
    plt.show()







if __name__ == '__main__':
    b = NASCifar10A(data_dir='nas_benchmarks/')

    # loading the model
    cfg = configs[4]
    model = Model(input_dim=5, hidden_dim=128, latent_dim=16,
                   num_hops=5, num_mlp_layers=2, dropout=0.3, **cfg['GAE']).cuda()
    dir_name = 'pretrained/dim-16/model-nasbench101.pt'
    model.load_state_dict(torch.load(dir_name)['model_state'])
    model.eval()

    EditDistance2L2Distance = random_walk(b)
    distance_compare_plot(EditDistance2L2Distance)



