import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

def plot_cdf_comparison(cmap=plt.get_cmap("tab10")):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    final_test_regret_rd_nas101 = []
    final_test_regret_re_nas101 = []
    final_test_regret_rl_nas101 = []
    final_test_regret_bohb_nas101 = []
    final_test_regret_rl_supervised = []
    final_test_regret_bo_supervised = []
    final_test_regret_rl_arch2vec = []
    final_test_regret_bo_arch2vec = []

    for i in range(1, 501):
        f_name = 'saved_logs/discrete/random_search/run_{}_nas_cifar10a_{}.json'.format(i, 20000)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for ind, t in enumerate(data['runtime']):
            if t > 1e6:
                final_test_regret_rd_nas101.append(data['regret_test'][ind])
                break
        f.close()

    for i in range(1, 501):
        f_name = 'saved_logs/discrete/regularized_evolution/run_{}_nas_cifar10a_{}.json'.format(i, 3500)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for ind, t in enumerate(data['runtime']):
            if t > 1e6:
                final_test_regret_re_nas101.append(data['regret_test'][ind])
                break
        f.close()

    for i in range(1, 501):
        f_name = 'saved_logs/discrete/rl/run_{}_nas_cifar10a_{}.json'.format(i, 3670)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for ind, t in enumerate(data['runtime']):
            if t > 1e6:
                final_test_regret_rl_nas101.append(data['regret_test'][ind])
                break
        f.close()

    for i in range(1, 501):
        f_name = 'saved_logs/discrete/bohb/run_{}_nas_cifar10a_{}.json'.format(i, 1000)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for ind, t in enumerate(data['runtime']):
            if t > 1e6:
                final_test_regret_bohb_nas101.append(data['regret_test'][ind])
                break
        f.close()

    for i in range(1, 501):
        f_name = 'saved_logs/rl/dim16/nasbench101_supervised_search_logs/run_{}_supervised_rl.json'.format(i)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for ind, t in enumerate(data['runtime']):
            if t > 1e6:
                final_test_regret_rl_supervised.append(data['regret_test'][ind])
                break
        f.close()

    for i in range(1, 501):
        f_name = 'saved_logs/bo/dim16/nasbench101_supervised_search_logs/run_{}_supervised_bo.json'.format(i)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for ind, t in enumerate(data['runtime']):
            if t > 1e6:
                final_test_regret_bo_supervised.append(data['regret_test'][ind])
                break
        f.close()

    for i in range(1, 501):
        f_name = 'saved_logs/rl/dim16/nasbench101_search_logs/run_{}_arch2vec-model-vae-nasbench-101.json'.format(i)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for ind, t in enumerate(data['runtime']):
            if t > 1e6:
                final_test_regret_rl_arch2vec.append(data['regret_test'][ind])
                break
        f.close()

    for i in range(1, 501):
        f_name = 'saved_logs/bo/dim16/nasbench101_search_logs/run_{}_arch2vec-model-vae-nasbench-101.json'.format(i)
        if not os.path.exists(f_name):
            continue
        f = open(f_name)
        data = json.load(f)
        for ind, t in enumerate(data['runtime']):
            if t > 1e6:
                final_test_regret_bo_arch2vec.append(data['regret_test'][ind])
                break
        f.close()


    plt_name_rd_nas101 = '{}: {}'.format('Discrete', 'Random Search')
    plt_name_re_nas101 = '{}: {}'.format('Discrete', 'Regularized Evolution')
    plt_name_rl_nas101 = '{}: {}'.format('Discrete', 'REINFORCE')
    plt_name_bohb_nas101 = '{}: {}'.format('Discrete', 'BOHB')
    plt_name_rl_supervised = '{}: {}'.format('Supervised', 'REINFORCE')
    plt_name_bo_supervised = '{}: {}'.format('Supervised', 'Bayesian Optimization')
    plt_name_rl_arch2vec = '{}: {}'.format('arch2vec', 'REINFORCE')
    plt_name_bo_arch2vec = '{}: {}'.format('arch2vec', 'Bayesian Optimization')

    plt.hist(final_test_regret_rd_nas101, bins=10, range=[8e-4, 1.2e-2], normed=True, cumulative=True, histtype='step', linestyle='--', color=cmap(1), lw=2, label=plt_name_rd_nas101)
    plt.hist(final_test_regret_re_nas101, bins=10, range=[8e-4, 1.2e-2], normed=True, cumulative=True, histtype='step', linestyle='--', lw=2.0, color=cmap(4), label=plt_name_re_nas101)
    plt.hist(final_test_regret_rl_nas101, bins=10, range=[8e-4, 1.2e-2], normed=True, cumulative=True, histtype='step', linestyle='--', lw=2.0, color=cmap(6), label=plt_name_rl_nas101)
    plt.hist(final_test_regret_bohb_nas101, bins=10, range=[8e-4, 1.2e-2], normed=True, cumulative=True, histtype='step', linestyle='--', lw=2.0, color=cmap(5), label=plt_name_bohb_nas101)
    plt.hist(final_test_regret_rl_supervised, bins=10, range=[8e-4, 1.2e-2], normed=True, cumulative=True, histtype='step', linestyle='-.', lw=2.0, color=cmap(7), label=plt_name_rl_supervised)
    plt.hist(final_test_regret_bo_supervised, bins=10, range=[8e-4, 1.2e-2], normed=True, cumulative=True, histtype='step', linestyle='-.', lw=2.0, color=cmap(9), label=plt_name_bo_supervised)
    plt.hist(final_test_regret_rl_arch2vec, bins=10, range=[8e-4, 1.2e-2], normed=True, cumulative=True, histtype='step', linestyle='-.', lw=2.0, color=cmap(0), label=plt_name_rl_arch2vec)
    plt.hist(final_test_regret_bo_arch2vec, bins=10, range=[8e-4, 1.2e-2], normed=True, cumulative=True, histtype='step', linestyle='-.', lw=2.0, color=cmap(3), label=plt_name_bo_arch2vec)
    fix_hist_step_vertical_line_at_end(ax)


    ax.set_xscale('log')
    ax.set_xlabel('final test regret', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    ax.legend(prop={"size":8}, handles=new_handles, labels=labels, loc='upper left')


    plt.show()

if __name__ == '__main__':
    plot_cdf_comparison()
