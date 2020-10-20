import json
import torch
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.getcwd())
from darts.cnn.genotypes import Genotype
from darts.cnn.model import NetworkImageNet as Network
from thop import profile

def process(geno):
    for i, item in enumerate(geno):
        geno[i] = tuple(geno[i])
    return geno

def transform_operations(ops):
    transform_dict = {'c_k-2': 0, 'c_k-1': 1, 'none': 2, 'max_pool_3x3': 3, 'avg_pool_3x3': 4, 'skip_connect': 5,
                      'sep_conv_3x3': 6, 'sep_conv_5x5': 7, 'dil_conv_3x3': 8, 'dil_conv_5x5': 9, 'output': 10}

    ops_array = np.zeros([11, 11], dtype='int8')
    for row, op in enumerate(ops):
        ops_array[row, op] = 1
    return ops_array

def sample_arch():
    num_ops = len(OPS)
    normal = []
    normal_name = []
    for i in range(NUM_VERTICES):
        ops = np.random.choice(range(num_ops), NUM_VERTICES)
        nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
        normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
        normal_name.extend([(str(nodes_in_normal[0]), OPS[ops[0]]), (str(nodes_in_normal[1]), OPS[ops[1]])])

    return (normal), (normal_name)


def build_mat_encoding(normal, normal_name, counter):
    adj = torch.zeros(11, 11)
    ops = torch.zeros(11, 11)
    block_0 = (normal[0], normal[1])
    prev_b0_n1, prev_b0_n2 = block_0[0][0], block_0[1][0]
    prev_b0_o1, prev_b0_o2 = block_0[0][1], block_0[1][1]

    block_1 = (normal[2], normal[3])
    prev_b1_n1, prev_b1_n2 = block_1[0][0], block_1[1][0]
    prev_b1_o1, prev_b1_o2 = block_1[0][1], block_1[1][1]

    block_2 = (normal[4], normal[5])
    prev_b2_n1, prev_b2_n2 = block_2[0][0], block_2[1][0]
    prev_b2_o1, prev_b2_o2 = block_2[0][1], block_2[1][1]

    block_3 = (normal[6], normal[7])
    prev_b3_n1, prev_b3_n2 = block_3[0][0], block_3[1][0]
    prev_b3_o1, prev_b3_o2 = block_3[0][1], block_3[1][1]

    adj[2][-1] = 1
    adj[3][-1] = 1
    adj[4][-1] = 1
    adj[5][-1] = 1
    adj[6][-1] = 1
    adj[7][-1] = 1
    adj[8][-1] = 1
    adj[9][-1] = 1

    # B0
    adj[prev_b0_n1][2] = 1
    adj[prev_b0_n2][3] = 1

    # B1
    if prev_b1_n1 == 2:
        adj[2][4] = 1
        adj[3][4] = 1
    else:
        adj[prev_b1_n1][4] = 1

    if prev_b1_n2 == 2:
        adj[2][5] = 1
        adj[3][5] = 1
    else:
        adj[prev_b1_n2][5] = 1

    # B2
    if prev_b2_n1 == 2:
        adj[2][6] = 1
        adj[3][6] = 1
    elif prev_b2_n1 == 3:
        adj[4][6] = 1
        adj[5][6] = 1
    else:
        adj[prev_b2_n1][6] = 1

    if prev_b2_n2 == 2:
        adj[2][7] = 1
        adj[3][7] = 1
    elif prev_b2_n2 == 3:
        adj[4][7] = 1
        adj[5][7] = 1
    else:
        adj[prev_b2_n2][7] = 1

    # B3
    if prev_b3_n1 == 2:
        adj[2][8] = 1
        adj[3][8] = 1
    elif prev_b3_n1 == 3:
        adj[4][8] = 1
        adj[5][8] = 1
    elif prev_b3_n1 == 4:
        adj[6][8] = 1
        adj[7][8] = 1
    else:
        adj[prev_b3_n1][8] = 1

    if prev_b3_n2 == 2:
        adj[2][9] = 1
        adj[3][9] = 1
    elif prev_b3_n2 == 3:
        adj[4][9] = 1
        adj[5][9] = 1
    elif prev_b3_n2 == 4:
        adj[6][9] = 1
        adj[7][9] = 1
    else:
        adj[prev_b3_n2][9] = 1

    ops[0][0] = 1
    ops[1][1] = 1
    ops[-1][-1] = 1
    ops[2][prev_b0_o1+2] = 1
    ops[3][prev_b0_o2+2] = 1
    ops[4][prev_b1_o1+2] = 1
    ops[5][prev_b1_o2+2] = 1
    ops[6][prev_b2_o1+2] = 1
    ops[7][prev_b2_o2+2] = 1
    ops[8][prev_b3_o1+2] = 1
    ops[9][prev_b3_o2+2] = 1

    #print("adj encoding: \n{} \n".format(adj.int()))
    #print("ops encoding: \n{} \n".format(ops.int()))

    label = torch.argmax(ops, dim=1)

    fingerprint = graph_util.hash_module(adj.int().numpy(), label.int().numpy().tolist())
    if fingerprint not in buckets:
        normal_cell = [(item[1], int(item[0])) for item in normal_name]
        reduce_cell = normal_cell.copy()
        genotype = Genotype(normal=normal_cell, normal_concat=[2, 3, 4, 5], reduce=reduce_cell, reduce_concat=[2, 3, 4, 5])
        model = Network(48, 1000, 14, False, genotype).cuda()
        input = torch.randn(1, 3, 224, 224).cuda()
        macs, params = profile(model, inputs=(input, ))
        if macs < 6e8:
            counter += 1
            print("counter: {}, flops: {}, params: {}".format(counter, macs, params))
            buckets[fingerprint] = (adj.numpy().astype('int8').tolist(), label.numpy().astype('int8').tolist(), (normal_name))

    if counter > 0 and counter % 1e5 == 0:
        with open('data/data_darts_counter{}.json'.format(counter), 'w') as f:
            json.dump(buckets, f)

    return counter

if __name__ == '__main__':
    from nasbench.lib import graph_util
    OPS = ['none',
           'max_pool_3x3',
           'avg_pool_3x3',
           'skip_connect',
           'sep_conv_3x3',
           'sep_conv_5x5',
           'dil_conv_3x3',
           'dil_conv_5x5'
           ]
    NUM_VERTICES = 4
    INPUT_1 = 'c_k-2'
    INPUT_2 = 'c_k-1'
    logging.basicConfig(filename='darts_preparation.log')

    buckets = {}
    counter = 0
    while counter <= 6e5:
        normal, normal_name = sample_arch()
        counter = build_mat_encoding(normal, normal_name, counter)
