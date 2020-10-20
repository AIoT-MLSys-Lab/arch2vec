import json
import matplotlib.pyplot as plt

def plot_over_time_comparison(cmap=plt.get_cmap("tab10")):
    fig = plt.figure()
    ax_test = fig.add_subplot(1, 1, 1)

    f_random_search = open('results/Random-Search-Encoding-A.json')
    f_regularized_evolution = open('results/Regularized-Evolution-Encoding-A.json')
    f_reinforce_search = open('results/Reinforce-Search-Encoding-A.json')
    f_bohb_search = open('results/BOHB-Search-Encoding-A.json')
    f_reinforce_search_arch2vec = open('results/RL-arch2vec-model-vae-nasbench-101.json')
    f_bo_search_arch2vec = open('results/BO-arch2vec-model-vae-nasbench-101.json')
    f_reinforce_search_supervised = open('results/RL-supervised-nasbench-101.json')
    f_bo_search_supervised = open('results/BO-supervised-nasbench-101.json')
    result_random_search = json.load(f_random_search)
    result_regularized_evolution = json.load(f_regularized_evolution)
    result_reinforce_search = json.load(f_reinforce_search)
    result_bohb_search = json.load(f_bohb_search)
    results_reinforce_search_arch2vec = json.load(f_reinforce_search_arch2vec)
    results_bo_search_arch2vec = json.load(f_bo_search_arch2vec)
    results_reinforce_search_supervised = json.load(f_reinforce_search_supervised)
    results_bo_search_supervised = json.load(f_bo_search_supervised)
    f_random_search.close()
    f_regularized_evolution.close()
    f_reinforce_search.close()
    f_bohb_search.close()
    f_reinforce_search_arch2vec.close()
    f_bo_search_arch2vec.close()
    f_reinforce_search_supervised.close()
    f_bo_search_supervised.close()

    ax_test.plot(result_random_search['time_plot'], result_random_search['test_plot'],  linestyle='-.', marker='^', markevery=1e3,  color=cmap(1), lw=2, markersize=4, label='{}: {}'.format('Discrete', 'Random Search'))
    ax_test.plot(result_regularized_evolution['time_plot'], result_regularized_evolution['test_plot'], linestyle='-.',  marker='s', markevery=1e3, color=cmap(4), lw=2, markersize=4, label='{}: {}'.format('Discrete', 'Regularized Evolution'))
    ax_test.plot(result_reinforce_search['time_plot'], result_reinforce_search['test_plot'], linestyle='-.', marker='.', markevery=1e3, color=cmap(6), lw=2, markersize=4, label='{}: {}'.format('Discrete', 'REINFORCE'))
    ax_test.plot(result_bohb_search['time_plot'], result_bohb_search['test_plot'] , linestyle='-.',  marker='*', markevery=1e3, color=cmap(5), lw=2, markersize=4, label='{}: {}'.format('Discrete', 'BOHB'))
    ax_test.plot(results_reinforce_search_supervised['time_plot'], results_reinforce_search_supervised['test_plot'], linestyle='--', marker='.', markevery=1e3, color=cmap(7), lw=2, markersize=4, label='{}: {}'.format('Supervised', 'REINFORCE'))
    ax_test.plot(results_bo_search_supervised['time_plot'], results_bo_search_supervised['test_plot'], linestyle='--',  marker='v', markevery=1e3, color=cmap(9), lw=2, markersize=4, label='{}: {}'.format('Supervised', 'Bayesian Optimization'))
    ax_test.plot(results_reinforce_search_arch2vec['time_plot'], results_reinforce_search_arch2vec['test_plot'], linestyle='-.',  marker='.', markevery=1e3, color=cmap(0), lw=2, markersize=4, label='{}: {}'.format('arch2vec', 'REINFORCE'))
    ax_test.plot(results_bo_search_arch2vec['time_plot'], results_bo_search_arch2vec['test_plot'], linestyle='-.',  marker='v', markevery=1e3, color=cmap(3), lw=2, markersize=4, label='{}: {}'.format('arch2vec', 'Bayesian Optimization'))

    ax_test.set_xscale('log')
    ax_test.set_yscale('log')
    ax_test.set_xlabel('estimated wall-clock time [s]', fontsize=12)
    ax_test.set_ylabel('test regret', fontsize=12)
    ax_test.legend(prop={"size":10})

    plt.show()

if __name__ == '__main__':
    plot_over_time_comparison()
