import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gin.models.mlp import MLP
from models.pretraining_nasbench101 import configs
from utils.utils import preprocessing
from torch.distributions import MultivariateNormal


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

class Env(object):
    def __init__(self, name, seed):
        self.name = name
        self.seed = seed
        self.dir_name = 'pretrained/dim-{}'.format(args.dim)
        self.visited = {}

    def extract_feature(self, model, X_adj, X_ops):
        model.eval()
        with torch.no_grad():
            window_size = 1024
            chunks = int(X_adj.shape[0] / window_size)
            features = []
            if X_adj.shape[0] % window_size > 0:
                chunks += 1
            X_adj_split = torch.split(X_adj, window_size, dim=0)
            X_ops_split = torch.split(X_ops, window_size, dim=0)
            for i in range(chunks):
                features.append(model._encoder(X_ops_split[i].cuda(), X_adj_split[i].cuda()))
        return torch.cat(features, dim=0).mean(dim=1).squeeze(1)

    def get_init_state(self, X_adj, X_ops, Y, Y_test, training_time):
        random.seed(args.seed)
        rand_indices = random.randint(0, X_adj.shape[0])
        self.visited[rand_indices] = True
        return X_adj[rand_indices], X_ops[rand_indices], Y[rand_indices], Y_test[rand_indices], training_time[rand_indices]

    def step(self, action, features, Y_valid, Y_test, training_time):
        dist = torch.norm(features.cpu() - action.cpu(), dim=1)
        knn = (-1 * dist).topk(dist.shape[0])
        min_dist, min_idx = knn.values, knn.indices
        count = 0
        while True:
            if len(self.visited) == dist.shape[0]:
                print("cannot find in the dataset")
                exit()
            if min_idx[count].item() not in self.visited:
                self.visited[min_idx[count].item()] = True
                break
            count += 1
        return features[min_idx[count].item()], Y_valid[min_idx[count].item()].item(), \
               Y_test[min_idx[count].item()].item(), training_time[min_idx[count].item()].item()

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers):
        super(Net, self).__init__()
        self.num_layers = num_hops
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.fc = nn.Linear(self.hidden_dim, self.latent_dim)

    def _encoder(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj, x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        x = self.fc(x)
        return x

    def forward(self, ops, adj):
        return self._encoder(ops, adj)

class Policy(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, input):
        x = F.relu(self.fc1(input))
        out = self.fc2(x)
        return out

class Policy_LSTM(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(Policy_LSTM, self).__init__()
        self.lstm = torch.nn.LSTMCell(input_size=hidden_dim1, hidden_size=hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, hidden_dim1)
        self.saved_log_probs = []
        self.rewards = []
        self.hx = None
        self.cx = None

    def forward(self, input):
        if self.hx is None and self.cx is None:
            self.hx, self.cx = self.lstm(input)
        else:
            self.hx, self.cx = self.lstm(input, (self.hx, self.cx))
        mean = self.fc(self.hx)
        return mean

def select_action(state, policy):
    state = state.mean(dim=1).squeeze(1).squeeze(0)
    mean = policy(state.view(1, state.shape[0]))
    mvn = MultivariateNormal(mean, torch.eye(state.shape[0]).cuda())
    action = mvn.sample()
    policy.saved_log_probs.append(torch.mean(mvn.log_prob(action)))
    return action


def finish_episode(policy, optimizer_gin, optimizer_policy):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards:
        R = r + 0.8 * R
        returns.append(R)
    returns = torch.Tensor(policy.rewards)
    returns = returns - 0.95
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer_gin.zero_grad()
    optimizer_policy.zero_grad()
    policy_loss = torch.mean(torch.stack(policy_loss, dim=0))
    print("average reward: {}, policy loss: {}".format(sum(policy.rewards)/len(policy.rewards), policy_loss.item()))
    policy_loss.backward()
    optimizer_policy.step()
    optimizer_gin.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    policy.hx = None
    policy.cx = None


def reinforce_search(X_adj, X_ops, Y, Y_test, training_time, env, args):
    """implementation of supervised learning based RL search"""
    model = Net(input_dim=5, hidden_dim=128, latent_dim=args.dim, num_hops=5, num_mlp_layers=2).cuda()
    policy = Policy_LSTM(args.dim, 128).cuda()
    optimizer_gin = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_policy= optim.Adam(policy.parameters(), lr=1e-2)
    counter = 0
    BEST_VALID_ACC = 0.9505542318026224
    BEST_TEST_ACC = 0.943175752957662
    MAX_BUDGET = 1.5e6
    rt = 0
    adj, ops, y_valid, y_test, time = env.get_init_state(X_adj, X_ops, Y, Y_test, training_time)
    CURR_BEST_VALID = y_valid.item()
    CURR_BEST_TEST = y_test.item()
    test_trace = [CURR_BEST_TEST]
    valid_trace = [CURR_BEST_VALID]
    time_trace = [time.item()]
    while rt < MAX_BUDGET:
        features = env.extract_feature(model, X_adj, X_ops)
        print('feature extraction finished. the shape is: ', features.shape)
        model.train()
        for c in range(args.bs):
            adj, ops = adj.cuda(), ops.cuda()
            state = model._encoder(ops.unsqueeze(0), adj.unsqueeze(0))
            action = select_action(state, policy)
            state, reward, reward_test, time = env.step(action, features, Y, Y_test, training_time)
            policy.rewards.append(reward)
            counter += 1
            rt += time
            print('counter: {}, validation reward: {}, test reward: {}, time: {}'.format(counter, reward, reward_test, rt))

            if reward > CURR_BEST_VALID:
                CURR_BEST_VALID = reward
                CURR_BEST_TEST = reward_test

            valid_trace.append(float(BEST_VALID_ACC - CURR_BEST_VALID))
            test_trace.append(float(BEST_TEST_ACC - CURR_BEST_TEST))
            time_trace.append(rt)

            if rt >= MAX_BUDGET:
                break

        finish_episode(policy, optimizer_gin, optimizer_policy)

    res = dict()
    res['regret_validation'] = valid_trace
    res['regret_test'] = test_trace
    res['runtime'] = time_trace
    save_path = os.path.join(args.output_path, 'dim{}'.format(args.dim))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fh = open(os.path.join(save_path, 'run_{}_{}.json'.format(args.seed, 'supervised_rl')),'w')
    json.dump(res, fh)
    fh.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Supervised RL")
    parser.add_argument("--gamma", type=float, default=0.8, help="discount factor (default 0.99)")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--dim', type=int, default=16, help='feature dimension')
    parser.add_argument('--output_path', type=str, default='saved_logs/rl', help='rl')
    args = parser.parse_args()
    cfg = configs[args.cfg]
    X_adj, X_ops, Y, Y_test, training_time = extract_data('data/data.json')
    env = Env('REINFORCE', args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    reinforce_search(X_adj, X_ops, Y, Y_test, training_time, env, args)











































