import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_over_time_reinforce_search_arch2vec(name, cmap=plt.get_cmap("tab10")):
    length = []
    for i in range(1, 501):
        f_name = 'saved_logs/rl/dim16/run_{}_{}-model-nasbench101.json'.format(i, name)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        length.append(len(data['runtime']))
        f.close()

    data_avg = defaultdict(list)
    test_regret_avg = defaultdict(list)
    valid_regret_avg = defaultdict(list)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax_test = fig.add_subplot(1, 2, 2)
    for i in range(1, 501):
        f_name = 'saved_logs/rl/dim16/run_{}_{}-model-nasbench101.json'.format(i, name)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for idx in range(min(length)):
            data_avg[idx].append(data['runtime'][idx])
            valid_regret_avg[idx].append(data['regret_validation'][idx])
            test_regret_avg[idx].append(data['regret_test'][idx])
        f.close()

    time_plot = []
    valid_plot = []
    test_plot = []
    for idx in range(min(length)):
        if sum(data_avg[idx]) / len(data_avg[idx]) > 1e6:
            continue
        time_plot.append(sum(data_avg[idx]) / len(data_avg[idx]))
        valid_plot.append(sum(valid_regret_avg[idx]) / len(valid_regret_avg[idx]))
        test_plot.append(sum(test_regret_avg[idx]) / len(test_regret_avg[idx]))

    ax.plot(time_plot, valid_plot, color=cmap(6), lw=2, label='{}: {}'.format('arch2vec', 'RL'))
    ax_test.plot(time_plot, test_plot, '--', color=cmap(6), lw=2, label='{}: {}'.format('arch2vec', 'RL'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('estimated wall-clock time [s]')
    ax.set_ylabel('validation regret')
    ax.legend()
    ax_test.set_xscale('log')
    ax_test.set_yscale('log')
    ax_test.set_xlabel('estimated wall-clock time [s]')
    ax_test.set_ylabel('test regret')
    ax_test.legend()

    save_data = {'time_plot': time_plot, 'valid_plot': valid_plot, 'test_plot': test_plot}
    with open('results/{}-{}-nasbench-101.json'.format('RL', name), 'w') as f_w:
        json.dump(save_data, f_w)

    plt.show()

if __name__ == '__main__':
    name = 'arch2vec'
    plot_over_time_reinforce_search_arch2vec(name)

