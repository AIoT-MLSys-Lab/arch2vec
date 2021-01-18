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
from utils.utils import load_json, preprocessing
from models.model import Model
from torch.distributions import MultivariateNormal

class Env(object):
    def __init__(self, name, seed, cfg, data_path=None, save=False):
        self.name = name
        self.seed = seed
        self.dir_name = 'pretrained/dim-{}'.format(args.latent_dim)
        print('Save file director is {}'.format(self.dir_name))
        self.visited = {}
        self.features = []
        self.embedding = {}
        self._reset(data_path, save)

    def _reset(self, data_path, save):
        if not save:
            print("extract arch2vec embedding table...")
            dataset = load_json(data_path)
            self.model = Model(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                                num_hops=args.hops, num_mlp_layers=args.mlps, dropout=args.dropout, **cfg['GAE']).cuda()
            model_ckpt_path = os.path.join(self.dir_name, '{}'.format(args.model_path))
            if not os.path.exists(model_ckpt_path):
                print("File {} does not exist.".format(model_ckpt_path))
                exit()
            self.model.load_state_dict(torch.load(model_ckpt_path)['model_state'])
            self.model.eval()
            print("length of the dataset: {}".format(len(dataset)))
            self.f_path = os.path.join(self.dir_name, '{}-arch2vec.pt'.format(args.dataset_name))
            if os.path.exists(self.f_path):
                print('ATTENTION!!! {} is already saved.'.format(self.f_path))
                exit()
            print('save to {} ...'.format(self.f_path))
            for ind in range(len(dataset)):
                adj = torch.Tensor(dataset[str(ind)]['module_adjacency']).unsqueeze(0).cuda()
                ops = torch.Tensor(dataset[str(ind)]['module_operations']).unsqueeze(0).cuda()
                adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
                test_acc = dataset[str(ind)]['test_accuracy']
                valid_acc = dataset[str(ind)]['validation_accuracy']
                other_info = {'valid_accuracy_avg':dataset[str(ind)]['validation_accuracy_avg'],
                              'test_accuracy_avg':dataset[str(ind)]['test_accuracy_avg']}
                time = dataset[str(ind)]['training_time']
                x, _ = self.model._encoder(ops, adj)
                self.embedding[ind] = {'feature': x.mean(dim=1).squeeze(0).cpu(), 'valid_accuracy': float(valid_acc), 'test_accuracy': float(test_acc),
                                       'time': float(time), 'other_info':other_info}
            torch.save(self.embedding, self.f_path)
            print("finished arch2vec extraction")
            exit()
        else:
            self.f_path = os.path.join(self.dir_name, '{}-arch2vec.pt'.format(args.dataset_name))
            print("load pretrained arch2vec in path: {}".format(self.f_path))
            self.embedding = torch.load(self.f_path)
            random.seed(args.seed)
            random.shuffle(self.embedding)
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
        return self.features[rand_indices], self.embedding[rand_indices]['valid_accuracy']/100.0, self.embedding[rand_indices]['test_accuracy']/100.0, self.embedding[rand_indices]['time'], self.embedding[rand_indices]['other_info']

    def step(self, action):
        """
        action: 1 x dim
        self.features. N x dim
        """
        dist = torch.norm(self.features - action.cpu(), dim=1)
        knn = (-1 * dist).topk(dist.shape[0])
        min_dist, min_idx = knn.values, knn.indices
        count = 1
        while True:
            if len(self.visited) == dist.shape[0]:
                print("CANNOT FIND IN THE ENTIRE DATASET !!!")
                exit()
            if min_idx[count].item() not in self.visited:
                self.visited[min_idx[count].item()] = True
                break
            count += 1

        return self.features[min_idx[count].item()], self.embedding[min_idx[count].item()]['valid_accuracy']/100.0, self.embedding[min_idx[count].item()]['test_accuracy']/100.0, self.embedding[min_idx[count].item()]['time'],  self.embedding[min_idx[count].item()]['other_info']


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
        out = self.fc(self.hx)
        return out



def select_action(state, policy):
    """
     MVN based action selection.
    :param state: 1 x dim
    :param policy: policy network
    :return: selected action: 1 x dim
    """
    out = policy(state.view(1, state.shape[0]))
    mvn = MultivariateNormal(out, 1.0*torch.eye(state.shape[0]).cuda())
    action = mvn.sample()
    policy.saved_log_probs.append(torch.mean(mvn.log_prob(action)))
    return action


def finish_episode(policy, optimizer, baseline):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards:
        R = r + args.gamma * R
        returns.append(R)
    returns = torch.Tensor(policy.rewards)
    val, indices = torch.sort(returns)
    print("sorted validation reward:", val)
    returns = returns - baseline
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.mean(torch.stack(policy_loss, dim=0))
    print("average reward: {}, policy loss: {}".format(sum(policy.rewards)/len(policy.rewards), policy_loss.item()))
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:] # to avoid active learning with increasing pool size
    del policy.saved_log_probs[:]
    policy.hx = None
    policy.cx = None



def reinforce_search(env):
    policy = Policy_LSTM(args.latent_dim, 128).cuda()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    counter = 0
    rt = 0
    state, _, _, time, _ = env.get_init_state()
    CURR_BEST_VALID = 0
    CURR_BEST_TEST = 0
    CURR_BEST_INFO = None
    test_trace = []
    valid_trace = []
    time_trace = []
    while rt < args.MAX_BUDGET:
        for c in range(args.bs):
            state = state.cuda()
            action = select_action(state, policy)
            state, reward, reward_test, time, other_info = env.step(action)
            policy.rewards.append(reward)
            counter += 1
            rt += time
            print('counter: {}, validation reward: {}, test reward: {}, time: {}'.format(counter, reward, reward_test, rt))

            if reward > CURR_BEST_VALID:
                CURR_BEST_VALID = reward
                CURR_BEST_TEST = reward_test
                CURR_BEST_INFO =  other_info

            valid_trace.append(float(CURR_BEST_VALID))
            test_trace.append(float(CURR_BEST_TEST))
            time_trace.append(rt)

            if rt >= args.MAX_BUDGET:
                break

        finish_episode(policy, optimizer, args.baseline)

    res = dict()
    res['regret_validation'] = valid_trace
    res['regret_test'] = test_trace
    res['runtime'] = time_trace
    save_path = os.path.join(args.output_path, 'dim{}'.format(args.latent_dim))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('save to {}'.format(save_path))
    print('Current Best Valid {}, Test {}'.format(CURR_BEST_VALID, CURR_BEST_TEST))
    data_dict = {'val_acc':float(CURR_BEST_VALID), 'test_acc':float(CURR_BEST_TEST),
                 'val_acc_avg': float(CURR_BEST_INFO['valid_accuracy_avg']),
                 'test_acc_avg': float(CURR_BEST_INFO['test_accuracy_avg'])}
    save_dir = os.path.join(save_path, 'nasbench201_{}_run_{}_full.json'.format(args.dataset_name, args.seed))
    with open(save_dir, 'w') as f:
        json.dump(data_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="reinforce search")
    parser.add_argument("--gamma", type=float, default=0.8, help="discount factor (default 0.99)")
    parser.add_argument('--baseline', type=float, default = 0.4, help='baseline value in rl')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=16, help='batch size for trajectory predictions, default=16')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--output_path', type=str, default='rl', help='rl/bo (default: rl)')
    parser.add_argument('--saved_arch2vec', action="store_true", default=False)
    parser.add_argument('--dataset_name', type=str, default='cifar10_valid_converged', help='Select from | cifar100 | ImageNet16_120 | cifar10_valid | cifar10_valid_converged')
    parser.add_argument('--model_path', type=str, default='model-nasbench201', help='The pretrained checkpoint to be loaded')
    parser.add_argument('--EMA_momentum', type=float, default=0.95, help='EMA momentum for reward baseline')
    parser.add_argument('--MAX_BUDGET', type=float, default = 12000, help='The budget in seconds')
    parser.add_argument('--input_dim', type=int, default=7)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    args = parser.parse_args()
    cfg = configs[args.cfg]
    env = Env('Reinforce', args.seed, cfg, data_path='data/{}.json'.format(args.dataset_name), save=args.saved_arch2vec)
    torch.manual_seed(args.seed)
    reinforce_search(env)
