import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.pretraining_nasbench101 import configs
from utils.utils import load_json, preprocessing, one_hot_darts
from preprocessing.gen_isomorphism_graphs import process
from models.model import Model
from torch.distributions import MultivariateNormal
from darts.cnn.train_search import Train


class Env(object):
    def __init__(self, name, seed, cfg, data_path=None, save=False):
        self.name = name
        self.seed = seed
        self.model = Model(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.dim,
                       num_hops=args.hops, num_mlp_layers=args.mlps, dropout=args.dropout, **cfg['GAE']).cuda()
        self.dir_name = 'pretrained/dim-{}'.format(args.dim)
        if not os.path.exists(os.path.join(self.dir_name, 'model-darts.pt')):
            exit()
        self.model.load_state_dict(torch.load(os.path.join(self.dir_name, 'model-darts.pt').format(args.dim))['model_state'])
        self.visited = {}
        self.features = []
        self.genotype = []
        self.embedding = {}
        self._reset(data_path, save)

    def _reset(self, data_path, save):
        if not save:
            print("extract arch2vec on DARTS search space ...")
            dataset = load_json(data_path)
            print("length of the dataset: {}".format(len(dataset)))
            self.f_path = os.path.join(self.dir_name, 'arch2vec-darts.pt')
            if os.path.exists(self.f_path):
                print('{} is already saved'.format(self.f_path))
                exit()
            print('save to {}'.format(self.f_path))
            counter = 0
            self.model.eval()
            for k, v in dataset.items():
                adj = torch.Tensor(v[0]).unsqueeze(0).cuda()
                ops = torch.Tensor(one_hot_darts(v[1])).unsqueeze(0).cuda()
                adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
                with torch.no_grad():
                    x, _ = self.model._encoder(ops, adj)
                    self.embedding[counter] = {'feature': x.squeeze(0).mean(dim=0).cpu(), 'genotype': process(v[2])}
                print("{}/{}".format(counter, len(dataset)))
                counter += 1
            torch.save(self.embedding, self.f_path)
            print("finished arch2vec extraction")
            exit()
        else:
            self.f_path = os.path.join(self.dir_name, 'arch2vec-darts.pt')
            print("load arch2vec from: {}".format(self.f_path))
            self.embedding = torch.load(self.f_path)
            for ind in range(len(self.embedding)):
                self.features.append(self.embedding[ind]['feature'])
                self.genotype.append(self.embedding[ind]['genotype'])
            self.features = torch.stack(self.features, dim=0)
            print('loading finished. pretrained embeddings shape: {}'.format(self.features.shape))


    def get_init_state(self):
        """
        :return: 1 x dim
        """
        rand_indices = random.randint(0, self.features.shape[0])
        self.visited[rand_indices] = True
        return self.features[rand_indices], self.genotype[rand_indices]

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
                print("CANNOT FIND IN THE DATASET!")
                exit()
            if min_idx[count].item() not in self.visited:
                self.visited[min_idx[count].item()] = True
                break
            count += 1

        return self.features[min_idx[count].item()], self.genotype[min_idx[count].item()]


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
    :return: selected action: 1 x dim
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
        R = r + args.gamma * R
        returns.append(R)
    returns = torch.Tensor(policy.rewards)
    val, indices = torch.sort(returns)
    print("sorted validation reward:", val)
    returns = returns - args.objective
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

def query(counter, seed, genotype, epochs):
    trainer = Train()
    rewards, rewards_test = trainer.main(counter, seed, genotype, epochs=epochs, train_portion=args.train_portion, save=args.logging_path)
    val_sum = 0
    for epoch, val_acc in rewards:
        val_sum += val_acc
    val_avg = val_sum / len(rewards)
    return val_avg / 100. , rewards_test[-1][-1] / 100.


def reinforce_search(env):
    """ implementation of arch2vec-RL on DARTS Search Space """
    policy = Policy_LSTM(args.dim, 128).cuda()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    counter = 0
    MAX_BUDGET = args.max_budgets
    state, genotype = env.get_init_state()
    CURR_BEST_VALID = 0
    CURR_BEST_TEST = 0
    CURR_BEST_GENOTYPE  = None
    test_trace = []
    valid_trace = []
    genotype_trace = []
    counter_trace = []
    while counter < MAX_BUDGET:
        for c in range(args.bs):
            state = state.cuda()
            action = select_action(state, policy)
            state, genotype = env.step(action)
            reward, reward_test = query(counter=counter, seed=args.seed, genotype=genotype, epochs=args.inner_epochs)
            policy.rewards.append(reward)
            counter += 1
            print('counter: {}, validation reward: {}, test reward: {}, genotype: {}'.format(counter, reward, reward_test, genotype))

            if reward > CURR_BEST_VALID:
                CURR_BEST_VALID = reward
                CURR_BEST_TEST = reward_test
                CURR_BEST_GENOTYPE = genotype

            valid_trace.append(float(CURR_BEST_VALID))
            test_trace.append(float(CURR_BEST_TEST))
            genotype_trace.append(CURR_BEST_GENOTYPE)
            counter_trace.append(counter)

            if counter >= MAX_BUDGET:
                break

        finish_episode(policy, optimizer)

    res = dict()
    res['validation_acc'] = valid_trace
    res['test_acc'] = test_trace
    res['genotype'] = genotype_trace
    res['counter'] = counter_trace
    save_path = os.path.join(args.output_path, 'dim{}'.format(args.dim))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('save to {}'.format(save_path))
    fh = open(os.path.join(save_path, 'run_{}_arch2vec_model_darts.json'.format(args.seed)), 'w')
    json.dump(res, fh)
    fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arch2vec-REINFORCE")
    parser.add_argument("--gamma", type=float, default=0.8, help="discount factor (default 0.99)")
    parser.add_argument("--seed", type=int, default=3, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--objective', type=float, default=0.95, help='rl baseline')
    parser.add_argument('--max_budgets', type=int, default=100, help='number of queries')
    parser.add_argument('--inner_epochs', type=int, default=50, help='inner loop epochs')
    parser.add_argument('--train_portion', type=float, default=0.9, help='train/validation split portion')
    parser.add_argument('--output_path', type=str, default='rl', help='rl/bo (default: rl)')
    parser.add_argument('--logging_path', type=str, default='', help='search logging path')
    parser.add_argument('--saved_arch2vec', action="store_true", default=False)
    parser.add_argument('--input_dim', type=int, default=11)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=16,
                        help='feature dimension (default: 16)')
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()
    cfg = configs[args.cfg]
    env = Env('REINFORCE', args.seed, cfg, data_path='data/data_darts_counter600000.json', save=args.saved_arch2vec)
    torch.manual_seed(args.seed)
    reinforce_search(env)
