import os
import sys
sys.path.insert(0, os.getcwd())
from pybnn.dngo import DNGO
import random
import argparse
import json
import torch
import numpy as np
from collections import defaultdict
from torch.distributions import Normal
import time


def load_arch2vec(embedding_path):
    embedding = torch.load(embedding_path)
    print('load pretrained arch2vec from {}'.format(embedding_path))
    random.seed(args.seed)
    random.shuffle(embedding)
    features = [embedding[ind]['feature'] for ind in range(len(embedding))]
    valid_labels = [embedding[ind]['valid_accuracy']/100.0 for ind in range(len(embedding))]
    test_labels = [embedding[ind]['test_accuracy']/100.0 for ind in range(len(embedding))]
    training_time = [embedding[ind]['time'] for ind in range(len(embedding))]
    other_info = [embedding[ind]['other_info'] for ind in range(len(embedding))]
    features = torch.stack(features, dim=0)
    valid_labels = torch.Tensor(valid_labels)
    test_labels = torch.Tensor(test_labels)
    training_time = torch.Tensor(training_time)
    print('loading finished. pretrained embeddings shape {}, and valid labels shape {}, and test labels shape {}'.format(features.shape, valid_labels.shape, test_labels.shape))
    return features, valid_labels, test_labels, training_time, other_info


def get_init_samples(features, valid_labels, test_labels, training_time, other_info, visited):
    np.random.seed(args.seed)
    init_inds = np.random.permutation(list(range(features.shape[0])))[:args.init_size]
    init_inds = torch.Tensor(init_inds).long()
    init_feat_samples = features[init_inds]
    init_valid_label_samples = valid_labels[init_inds]
    init_test_label_samples = test_labels[init_inds]
    init_time_samples = training_time[init_inds]
    print('='*20, init_inds)
    init_other_info_samples = [other_info[k] for k in init_inds]
    for idx in init_inds:
        visited[idx] = True
    return init_feat_samples, init_valid_label_samples, init_test_label_samples, init_time_samples, init_other_info_samples, visited


def propose_location(ei, features, valid_labels, test_labels, training_time, other_info, visited):
    k = args.batch_size
    print('remaining length of indices set:', len(features) - len(visited))
    indices = torch.argsort(ei)[-k:]
    ind_dedup = []
    # remove random sampled indices at each step
    for idx in indices:
        if idx not in visited:
            visited[idx] = True
            ind_dedup.append(idx)
    ind_dedup = torch.Tensor(ind_dedup).long()
    proposed_x, proposed_y_valid, proposed_y_test, proposed_time, propose_info = features[ind_dedup], valid_labels[ind_dedup], test_labels[ind_dedup], training_time[ind_dedup], [other_info[k] for k in ind_dedup]
    return proposed_x, proposed_y_valid, proposed_y_test, proposed_time, propose_info, visited


def expected_improvement_search(features, valid_labels, test_labels, training_time, other_info):
    """ implementation of expected improvement search given arch2vec.
    :param data_path: the pretrained arch2vec path.
    :return: features, labels
    """
    CURR_BEST_VALID = 0.
    CURR_BEST_TEST = 0.
    CURR_BEST_INFO = None
    MAX_BUDGET = args.MAX_BUDGET
    window_size = 200
    counter = 0
    rt = 0.
    visited = {}
    best_trace = defaultdict(list)

    features, valid_labels, test_labels, training_time = features.cpu().detach(), valid_labels.cpu().detach(), test_labels.cpu().detach(), training_time.cpu().detach()
    feat_samples, valid_label_samples, test_label_samples, time_samples, other_info_sampled, visited = get_init_samples(features, valid_labels, test_labels, training_time, other_info, visited)

    t_start = time.time()
    for feat, acc_valid, acc_test, t, o_info in zip(feat_samples, valid_label_samples, test_label_samples, time_samples, other_info_sampled):
        counter += 1
        rt += t.item()
        if acc_valid > CURR_BEST_VALID:
            CURR_BEST_VALID = acc_valid
            CURR_BEST_TEST = acc_test
            CURR_BEST_INFO = o_info
        best_trace['validation'].append(float(CURR_BEST_VALID))
        best_trace['test'].append(float(CURR_BEST_TEST))
        best_trace['time'].append(time.time() - t_start)
        best_trace['counter'].append(counter)

    while rt < MAX_BUDGET:
        print("feat_samples:", feat_samples.shape)
        print("valid label_samples:", valid_label_samples.shape)
        print("test label samples:", test_label_samples.shape)
        print("current best validation: {}".format(CURR_BEST_VALID))
        print("current best test: {}".format(CURR_BEST_TEST))
        print("rt: {}".format(rt))
        print(feat_samples.shape)
        print(valid_label_samples.shape)
        model = DNGO(num_epochs=100, n_units=128, do_mcmc=False, normalize_output=False)
        model.train(X=feat_samples.numpy(), y=valid_label_samples.view(-1).numpy(), do_optimize=True)
        print(model.network)
        m = []
        v = []
        chunks = int(features.shape[0] / window_size)
        if features.shape[0] % window_size > 0:
            chunks += 1
        features_split = torch.split(features, window_size, dim=0)
        for i in range(chunks):
            m_split, v_split = model.predict(features_split[i].numpy())
            m.extend(list(m_split))
            v.extend(list(v_split))
        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        u = (mean - torch.Tensor([1.0]).expand_as(mean)) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        feat_next, label_next_valid, label_next_test, time_next, info_next, visited = propose_location(ei, features, valid_labels, test_labels, training_time, other_info, visited)

        # add proposed networks to selected networks
        for feat, acc_valid, acc_test, t, o_info in zip(feat_next, label_next_valid, label_next_test, time_next, info_next):
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), acc_valid.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), acc_test.view(1, 1)), dim=0)
            counter += 1
            rt += t.item()
            if acc_valid > CURR_BEST_VALID:
                CURR_BEST_VALID = acc_valid
                CURR_BEST_TEST = acc_test
                CURR_BEST_INFO = o_info

            best_trace['acc_validation'].append(float( CURR_BEST_VALID))
            best_trace['acc_test'].append(float(CURR_BEST_TEST))
            best_trace['search_time'].append(time.time() - t_start) # The actual searching time
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
    print('Current Best Valid {}, Test {}'.format(CURR_BEST_VALID, CURR_BEST_TEST))
    data_dict = {'val_acc': float(CURR_BEST_VALID), 'test_acc': float(CURR_BEST_TEST),
                 'val_acc_avg': float(CURR_BEST_INFO['valid_accuracy_avg']),
                 'test_acc_avg': float(CURR_BEST_INFO['test_accuracy_avg'])}
    save_dir = os.path.join(save_path, 'nasbench201_{}_run_{}_full.json'.format(args.dataset_name, args.seed))
    with open(save_dir, 'w') as f:
        json.dump(data_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DNGO search for NB201")
    parser.add_argument("--gamma", type=float, default=0, help="discount factor (default 0.99)")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--dim', type=int, default=16, help='feature dimension')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--batch_size', type=int, default=1, help='acquisition samples')
    parser.add_argument('--output_path', type=str, default='saved_logs/bo', help='rl/gd/predictor/bo (default: bo)')
    parser.add_argument('--saved_arch2vec', action="store_true", default=True)

    parser.add_argument('--dataset_name', type=str, default='ImageNet16_120',
                        help='Select from | cifar100 | ImageNet16_120 | cifar10_valid | cifar10_valid_converged')
    parser.add_argument('--MAX_BUDGET', type=float, default=1200000, help='The budget in seconds')

    args = parser.parse_args()
    #reproducbility is good
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    embedding_path = 'pretrained/dim-{}/{}-arch2vec.pt'.format(args.dim, args.dataset_name)
    if not os.path.exists(embedding_path):
        exit()
    features, valid_labels, test_labels, training_time, other_info = load_arch2vec(embedding_path)
    expected_improvement_search(features, valid_labels, test_labels, training_time, other_info)




