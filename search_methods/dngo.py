import os
import sys
sys.path.insert(0, os.getcwd())
from pybnn.dngo import DNGO
import argparse
import json
import torch
import numpy as np
from collections import defaultdict
from torch.distributions import Normal


def load_arch2vec(embedding_path):
    embedding = torch.load(embedding_path)
    print('load arch2vec from {}'.format(embedding_path))
    ind_list = range(len(embedding))
    features = [embedding[ind]['feature'] for ind in ind_list]
    valid_labels = [embedding[ind]['valid_accuracy'] for ind in ind_list]
    test_labels = [embedding[ind]['test_accuracy'] for ind in ind_list]
    training_time = [embedding[ind]['time'] for ind in ind_list]
    features = torch.stack(features, dim=0)
    test_labels = torch.Tensor(test_labels)
    valid_labels = torch.Tensor(valid_labels)
    training_time = torch.Tensor(training_time)
    print('loading finished. pretrained embeddings shape {}'.format(features.shape))
    return features, valid_labels, test_labels, training_time


def get_init_samples(features, valid_labels, test_labels, training_time, visited):
    np.random.seed(args.seed)
    init_inds = np.random.permutation(list(range(features.shape[0])))[:args.init_size]
    init_inds = torch.Tensor(init_inds).long()
    init_feat_samples = features[init_inds]
    init_valid_label_samples = valid_labels[init_inds]
    init_test_label_samples = test_labels[init_inds]
    init_time_samples = training_time[init_inds]
    for idx in init_inds:
        visited[idx] = True
    return init_feat_samples, init_valid_label_samples, init_test_label_samples, init_time_samples, visited


def propose_location(ei, features, valid_labels, test_labels, training_time, visited):
    k = args.topk
    print('remaining length of indices set:', len(features) - len(visited))
    indices = torch.argsort(ei)[-k:]
    ind_dedup = []
    for idx in indices:
        if idx not in visited:
            visited[idx] = True
            ind_dedup.append(idx)
    ind_dedup = torch.Tensor(ind_dedup).long()
    proposed_x, proposed_y_valid, proposed_y_test, proposed_time = features[ind_dedup], valid_labels[ind_dedup], test_labels[ind_dedup], training_time[ind_dedup]
    return proposed_x, proposed_y_valid, proposed_y_test, proposed_time, visited


def expected_improvement_search():
    """ implementation of arch2vec-DNGO """
    BEST_TEST_ACC = 0.943175752957662 
    BEST_VALID_ACC = 0.9505542318026224
    CURR_BEST_VALID = 0.
    CURR_BEST_TEST = 0.
    MAX_BUDGET = 1.5e6
    window_size = 200
    counter = 0
    rt = 0.
    visited = {}
    best_trace = defaultdict(list)
    features, valid_labels, test_labels, training_time = load_arch2vec(os.path.join('pretrained/dim-{}'.format(args.dim), args.emb_path))
    features, valid_labels, test_labels, training_time = features.cpu().detach(), valid_labels.cpu().detach(), test_labels.cpu().detach(), training_time.cpu().detach()
    feat_samples, valid_label_samples, test_label_samples, time_samples, visited = get_init_samples(features, valid_labels, test_labels, training_time, visited)

    for feat, acc_valid, acc_test, t in zip(feat_samples, valid_label_samples, test_label_samples, time_samples):
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
        print("feat_samples:", feat_samples.shape)
        print("valid label_samples:", valid_label_samples.shape)
        print("test label samples:", test_label_samples.shape)
        print("current best validation: {}".format(CURR_BEST_VALID))
        print("current best test: {}".format(CURR_BEST_TEST))
        print("rt: {}".format(rt))
        print(feat_samples.shape)
        print(valid_label_samples.shape)
        model = DNGO(num_epochs=100, n_units=128, do_mcmc=False, normalize_output=False, rng=args.seed)
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
        u = (mean - torch.Tensor([0.95]).expand_as(mean)) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        feat_next, label_next_valid, label_next_test, time_next, visited = propose_location(ei, features, valid_labels, test_labels, training_time, visited)

        # add proposed networks to the pool
        for feat, acc_valid, acc_test, t in zip(feat_next, label_next_valid, label_next_test, time_next):
            if acc_valid > CURR_BEST_VALID:
                CURR_BEST_VALID = acc_valid
                CURR_BEST_TEST = acc_test
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), acc_valid.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), acc_test.view(1, 1)), dim=0)
            counter += 1
            rt += t.item()
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
    if args.emb_path.endswith('.pt'):
        s = args.emb_path[:-3]
    fh = open(os.path.join(save_path, 'run_{}_{}.json'.format(args.seed, s)),'w')
    json.dump(res, fh)
    fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arch2vec-DNGO")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--dim', type=int, default=16, help='feature dimension')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--topk', type=int, default=5, help='acquisition samples')
    parser.add_argument('--output_path', type=str, default='bo', help='bo')
    parser.add_argument('--emb_path', type=str, default='arch2vec.pt')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    expected_improvement_search()




