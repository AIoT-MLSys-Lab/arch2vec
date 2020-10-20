import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from pybnn.dngo_supervised import DNGO
import json
import argparse
from collections import defaultdict
from torch.distributions import Normal

def extract_data(dataset):
    with open(dataset) as f:
        data = json.load(f)
    X_adj = [torch.Tensor(data[str(ind)]['module_adjacency']) for ind in range(len(data))]
    X_ops = [torch.Tensor(data[str(ind)]['module_operations']) for ind in range(len(data))]
    Y = [data[str(ind)]['validation_accuracy'] for ind in range(len(data))]
    Y_test = [data[str(ind)]['test_accuracy'] for ind in range(len(data))]
    training_time = [data[str(ind)]['training_time'] for ind in range(len(data))]
    X_adj = torch.stack(X_adj, dim=0)
    X_ops = torch.stack(X_ops, dim=0)
    Y = torch.Tensor(Y)
    Y_test = torch.Tensor(Y_test)
    training_time = torch.Tensor(training_time)
    rand_ind = torch.randperm(X_ops.shape[0])
    X_adj = X_adj[rand_ind]
    X_ops = X_ops[rand_ind]
    Y = Y[rand_ind]
    Y_test = Y_test[rand_ind]
    training_time = training_time[rand_ind]
    print('loading finished. input adj shape {}, input ops shape {} and valid labels shape {}, and test labels shape {}'.format(X_adj.shape, X_ops.shape, Y.shape, Y_test.shape))
    return X_adj, X_ops, Y, Y_test, training_time

def get_init_samples(X_adj, X_ops, Y, Y_test, training_time, visited):
    np.random.seed(args.seed)
    init_inds = np.random.permutation(list(range(X_ops.shape[0])))[:args.init_size]
    init_inds = torch.Tensor(init_inds).long()
    init_x_adj_samples = X_adj[init_inds]
    init_x_ops_samples = X_ops[init_inds]
    init_valid_label_samples = Y[init_inds]
    init_test_label_samples = Y_test[init_inds]
    init_time_samples = training_time[init_inds]
    for idx in init_inds:
        visited[idx.item()] = True
    return init_x_adj_samples, init_x_ops_samples, init_valid_label_samples, init_test_label_samples, init_time_samples, visited


def propose_location(ei, X_adj, X_ops, valid_labels, test_labels, training_time, visited):
    k = args.topk
    count = 0
    print('remaining length of indices set:', len(X_adj) - len(visited))
    indices = torch.argsort(ei)
    ind_dedup = []
    # remove random sampled indices at each step
    for idx in reversed(indices):
        if count == k:
            break
        if idx.item() not in visited:
            visited[idx.item()] = True
            ind_dedup.append(idx.item())
            count += 1
    ind_dedup = torch.Tensor(ind_dedup).long()
    proposed_x_adj, proposed_x_ops, proposed_y_valid, proposed_y_test, proposed_time = X_adj[ind_dedup], X_ops[ind_dedup], valid_labels[ind_dedup], test_labels[ind_dedup], training_time[ind_dedup]
    return proposed_x_adj, proposed_x_ops, proposed_y_valid, proposed_y_test, proposed_time, visited


def supervised_encoding_search(X_adj, X_ops, Y, Y_test, training_time):
    """implementation of supervised learning based BO search"""
    BEST_TEST_ACC = 0.943175752957662
    BEST_VALID_ACC = 0.9505542318026224
    CURR_BEST_VALID = 0.
    CURR_BEST_TEST = 0.
    MAX_BUDGET = 1.5e6
    counter = 0
    rt = 0.
    best_trace = defaultdict(list)
    window_size = 512
    visited = {}
    X_adj_sample, X_ops_sample, Y_sample, Y_sample_test, time_sample, visited = get_init_samples(X_adj, X_ops, Y, Y_test, training_time, visited)

    for x_adj, x_ops, acc_valid, acc_test, t in zip(X_adj_sample, X_ops_sample, Y_sample, Y_sample_test, time_sample):
        counter += 1
        rt += t.item()
        if acc_valid > CURR_BEST_VALID:
            CURR_BEST_VALID = acc_valid
            CURR_BEST_TEST = acc_test
        best_trace['regret_validation'].append(float(BEST_VALID_ACC - CURR_BEST_VALID))
        best_trace['regret_test'].append(float(BEST_TEST_ACC - CURR_BEST_TEST))
        best_trace['time'].append(rt)
        best_trace['counter'].append(counter)

    while rt < MAX_BUDGET:
        print("data adjacent matrix samples:", X_adj_sample.shape)
        print("data operations matrix samples:", X_ops_sample.shape)
        print("valid label_samples:", Y_sample.shape)
        print("test label samples:", Y_sample_test.shape)
        print("current best validation: {}".format(CURR_BEST_VALID))
        print("current best test: {}".format(CURR_BEST_TEST))
        print("rt: {}".format(rt))
        model = DNGO(num_epochs=100, input_dim=5, hidden_dim=128, latent_dim=args.dim, num_hops=5, num_mlp_layers=2, do_mcmc=False, normalize_output=False)
        model.train(X_adj_sample.numpy(), X_ops_sample.numpy(), Y_sample.view(-1).numpy(), do_optimize=True)
        m = []
        v = []
        chunks = int(X_adj.shape[0] / window_size)
        if X_adj.shape[0] % window_size > 0:
            chunks += 1
        X_adj_split = torch.split(X_adj, window_size, dim=0)
        X_ops_split = torch.split(X_ops, window_size, dim=0)
        for i in range(chunks):
            inputs_adj = X_adj_split[i]
            inputs_ops = X_ops_split[i]
            m_split, v_split = model.predict(inputs_ops.numpy(), inputs_adj.numpy())
            m.extend(list(m_split))
            v.extend(list(v_split))
        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        u = mean - torch.Tensor([0.95]).expand_as(mean) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)

        X_adj_next, X_ops_next, label_next_valid, label_next_test, time_next, visited = propose_location(ei, X_adj, X_ops, Y, Y_test, training_time, visited)

        # add proposed networks to selected networks
        for x_adj, x_ops, acc_valid, acc_test, t in zip(X_adj_next, X_ops_next, label_next_valid, label_next_test, time_next):
            X_adj_sample = torch.cat((X_adj_sample, x_adj.view(1, 7, 7)), dim=0)
            X_ops_sample = torch.cat((X_ops_sample, x_ops.view(1, 7, 5)), dim=0)
            Y_sample = torch.cat((Y_sample.view(-1, 1), acc_valid.view(1, 1)), dim=0)
            Y_sample_test = torch.cat((Y_sample_test.view(-1, 1), acc_test.view(1, 1)), dim=0)
            counter += 1
            rt += t.item()
            if acc_valid > CURR_BEST_VALID:
                CURR_BEST_VALID = acc_valid
                CURR_BEST_TEST = acc_test

            best_trace['regret_validation'].append(float(BEST_VALID_ACC - CURR_BEST_VALID))
            best_trace['regret_test'].append(float(BEST_TEST_ACC - CURR_BEST_TEST))
            best_trace['time'].append(rt)
            best_trace['counter'].append(counter)

            if rt >= MAX_BUDGET:
                break

    res = dict()
    res['regret_validation'] = best_trace['regret_validation']
    res['regret_test'] = best_trace['regret_test']
    res['runtime'] = best_trace['time']
    res['counter'] = best_trace['counter']
    save_path = os.path.join(args.output_path, 'dim{}'.format(args.dim))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('save to {}'.format(save_path))
    fh = open(os.path.join(save_path, 'run_{}_{}.json'.format(args.seed, args.benchmark)), 'w')
    json.dump(res, fh)
    fh.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Supervised DNGO search")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--dim', type=int, default=16, help='feature dimension')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--topk', type=int, default=5, help='acquisition samples')
    parser.add_argument('--benchmark', type=str, default='supervised_dngo')
    parser.add_argument('--output_path', type=str, default='saved_logs/bo', help='rl/bo (default: bo)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data_path = 'data/data.json'
    X_adj, X_ops, Y, Y_test, training_time = extract_data(data_path)
    supervised_encoding_search(X_adj, X_ops, Y, Y_test, training_time)
