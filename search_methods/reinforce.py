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
from models.pretraining_nasbench101 import configs
from utils.utils import load_json, preprocessing
from models.model import Model
from torch.distributions import MultivariateNormal

class Env(object):
    def __init__(self, name, seed, emb_path, model_path, cfg, data_path=None, save=False):
        self.name = name
        self.model_path = model_path
        self.emb_path = emb_path
        self.seed = seed
        self.dir_name = 'pretrained/dim-{}'.format(args.dim)
        self.visited = {}
        self.features = []
        self.embedding = {}
        self._reset(data_path, save)

    def _reset(self, data_path, save):
        if not save:
            print("extract arch2vec from {}".format(os.path.join(self.dir_name, self.model_path)))
            if not os.path.exists(os.path.join(self.dir_name, self.model_path)):
                exit()
            dataset = load_json(data_path)
            self.model = Model(input_dim=5, hidden_dim=128, latent_dim=16, num_hops=5, num_mlp_layers=2, dropout=0, **cfg['GAE']).cuda()
            self.model.load_state_dict(torch.load(os.path.join(self.dir_name, self.model_path).format(args.dim))['model_state'])
            self.model.eval()
            with torch.no_grad():
                print("length of the dataset: {}".format(len(dataset)))
                self.f_path = os.path.join(self.dir_name, 'arch2vec-{}'.format(self.model_path))
                if os.path.exists(self.f_path):
                    print('{} is already saved'.format(self.f_path))
                    exit()
                print('save to {}'.format(self.f_path))
                for ind in range(len(dataset)):
                    adj = torch.Tensor(dataset[str(ind)]['module_adjacency']).unsqueeze(0).cuda()
                    ops = torch.Tensor(dataset[str(ind)]['module_operations']).unsqueeze(0).cuda()
                    adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
                    test_acc = dataset[str(ind)]['test_accuracy']
                    valid_acc = dataset[str(ind)]['validation_accuracy']
                    time = dataset[str(ind)]['training_time']
                    x,_ = self.model._encoder(ops, adj)
                    self.embedding[ind] = {'feature': x.squeeze(0).mean(dim=0).cpu(), 'valid_accuracy': float(valid_acc), 'test_accuracy': float(test_acc), 'time': float(time)}
                torch.save(self.embedding, self.f_path)
                print("finish arch2vec extraction")
                exit()
        else:
            self.f_path = os.path.join(self.dir_name, self.emb_path)
            print("load arch2vec from: {}".format(self.f_path))
            self.embedding = torch.load(self.f_path)
            for ind in range(len(self.embedding)):
                self.features.append(self.embedding[ind]['feature'])
            self.features = torch.stack(self.features, dim=0)
            print('loading finished. pretrained embeddings shape: {}'.format(self.features.shape))

    def get_init_state(self):
        """
        :return: 1 x dim
        """
        random.seed(args.seed)
        rand_indices = random.randint(0, self.features.shape[0])
        self.visited[rand_indices] = True
        return self.features[rand_indices], self.embedding[rand_indices]['valid_accuracy'],\
               self.embedding[rand_indices]['test_accuracy'], self.embedding[rand_indices]['time']

    def step(self, action):
        """
        action: 1 x dim
        self.features. N x dim
        """
        dist = torch.norm(self.features - action.cpu(), dim=1)
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

        return self.features[min_idx[count].item()], self.embedding[min_idx[count].item()]['valid_accuracy'], \
               self.embedding[min_idx[count].item()]['test_accuracy'], self.embedding[min_idx[count].item()]['time']


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
    """
     MVN based action selection.
    :param state: 1 x dim
    :param policy: policy network
    :return: action: 1 x dim
    """
    mean = policy(state.view(1, state.shape[0]))
    mvn = MultivariateNormal(mean, torch.eye(state.shape[0]).cuda())
    action = mvn.sample()
    policy.saved_log_probs.append(torch.mean(mvn.log_prob(action)))
    return action


def finish_episode(policy, optimizer):
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

    optimizer.zero_grad()
    policy_loss = torch.mean(torch.stack(policy_loss, dim=0))
    print("average reward: {}, policy loss: {}".format(sum(policy.rewards)/len(policy.rewards), policy_loss.item()))
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    policy.hx = None
    policy.cx = None


def reinforce_search(env, args):
    """ implementation of arch2vec-REINFORCE """
    policy = Policy_LSTM(args.dim, 128).cuda()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    counter = 0
    BEST_VALID_ACC = 0.9505542318026224
    BEST_TEST_ACC = 0.943175752957662
    MAX_BUDGET = 1.5e6
    rt = 0
    state, _, _, time = env.get_init_state()
    CURR_BEST_VALID = 0
    CURR_BEST_TEST = 0
    test_trace = []
    valid_trace = []
    time_trace = []
    while rt < MAX_BUDGET:
        for c in range(args.bs):
            state = state.cuda()
            action = select_action(state, policy)
            state, reward, reward_test, time = env.step(action)
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

        finish_episode(policy, optimizer)

    res = dict()
    res['regret_validation'] = valid_trace
    res['regret_test'] = test_trace
    res['runtime'] = time_trace
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
    parser = argparse.ArgumentParser(description="arch2vec-REINFORCE")
    parser.add_argument("--gamma", type=float, default=0, help="discount factor (default 0.99)")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--dim', type=int, default=7, help='feature dimension')
    parser.add_argument('--output_path', type=str, default='rl', help='rl/bo')
    parser.add_argument('--emb_path', type=str, default='arch2vec.pt')
    parser.add_argument('--model_path', type=str, default='model-nasbench-101.pt')
    parser.add_argument('--saved_arch2vec', action="store_true", default=False)
    args = parser.parse_args()
    cfg = configs[args.cfg]
    env = Env('REINFORCE', args.seed, args.emb_path, args.model_path, cfg, data_path='data/data.json', save=args.saved_arch2vec)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    reinforce_search(env, args)
