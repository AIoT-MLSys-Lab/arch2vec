import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import stats
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import os

result_path = 'saved_logs/predict_accuracy'
seed = [1, 10]
acc_th = 0.8

for s in seed:
    ## unsupervised
    un_pred_acc = np.load(os.path.join(result_path, 'dngo_unsupervised', 'pred_acc_seed{}.npy'.format(s)))
    un_test_acc = np.load(os.path.join(result_path, 'dngo_unsupervised', 'test_acc_seed{}.npy'.format(s)))
    idx0 = np.logical_and(un_test_acc > acc_th, un_pred_acc > acc_th)  # np.logical_and(un_pred_acc > th, un_test_acc > th)

    ## supervised
    sup_pred_acc = np.load(os.path.join(result_path, 'dngo_supervised', 'pred_acc_seed{}.npy'.format(s)))
    sup_test_acc = np.load(os.path.join(result_path, 'dngo_supervised', 'test_acc_seed{}.npy'.format(s)))
    idx1 = np.logical_and(sup_test_acc > acc_th, sup_pred_acc > acc_th)  # np.logical_and(sup_pred_acc > th, sup_test_acc > th)

    bins = np.linspace(0.8, 1, 301)

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(6, 3), sharey=True)

    ax0.plot([0.8, 1], [0.8, 1], 'yellowgreen', linewidth=2)
    ax1.plot([0.8, 1], [0.8, 1], 'yellowgreen', linewidth=2)

    H, xedges, yedges = np.histogram2d(un_test_acc[idx0], un_pred_acc[idx0], bins=bins)
    H = H.T
    Hm = np.ma.masked_where(H < 1, H)
    X, Y = np.meshgrid(xedges, yedges)
    palette = copy(plt.cm.viridis)
    palette.set_bad('w', 1.0)
    ax0.pcolormesh(X, Y, Hm, cmap=palette)

    H, xedges, yedges = np.histogram2d(sup_test_acc[idx1], un_pred_acc[idx1], bins=bins)
    H = H.T
    Hm = np.ma.masked_where(H < 1, H)
    X, Y = np.meshgrid(xedges, yedges)
    palette = copy(plt.cm.viridis)
    palette.set_bad('w', 1.0)
    ax1.pcolormesh(X, Y, Hm, cmap=palette)

    ax0.set_xlabel('Test Accuracy')
    ax0.set_ylabel('Predicted Accuracy')
    ax1.set_xlabel('Test Accuracy')

    ax0.set_xlim(0.8, 0.95)
    ax0.set_ylim(0.8, 0.95)
    ax1.set_xlim(0.8, 0.95)
    ax1.set_ylim(0.8, 0.95)

    ax0.set_yticks(ticks=[0.8, 0.85, 0.90, 0.95])
    ax0.set_xticks(ticks=[0.8, 0.85, 0.9])
    ax1.set_xticks(ticks=[0.8, 0.85, 0.9, 0.95])

    ax0.set_aspect('equal', 'box')
    ax1.set_aspect('equal', 'box')

    plt.subplots_adjust(wspace=0.05, top=0.9, bottom=0.1)
    plt.show()
    plt.savefig('compare_seed{}.png'.format(s), bbox_inches='tight')
    plt.close(fig=fig)


