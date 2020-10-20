import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils.random import sample_without_replacement
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import copy
import matplotlib.colors as colors
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.curdir)



def visualize2D(emb_path, out_path, seed=0, sample_num=1000, acc_dyn=(0.82, 0.92)):
    ## load embedding
    dataset = torch.load(emb_path)
    _, emb_name = os.path.split(emb_path)
    feature = []
    test_acc = []
    if sample_num <= 0 or sample_num >= len(dataset):
        sample_idx = range(len(dataset))
        sample_num = len(dataset)
    else:
        sample_idx = sample_without_replacement(len(dataset), sample_num, random_state=0)
    for j in tqdm(range(len(sample_idx)), desc='load feature'):
        i = sample_idx[j]
        feature.append(dataset[i]['feature'].detach().numpy())
        test_acc.append(dataset[i]['test_accuracy'])
    feature = np.stack(feature, axis=0)
    test_acc = np.stack(test_acc, axis=0)
    ## tsne reduces dim
    print('TSNE...')
    tsne = TSNE(random_state=seed)
    emb_feature = tsne.fit_transform(feature)
    emb_x = emb_feature[:, 0] / np.amax(np.abs(emb_feature[:, 0]))
    emb_y = emb_feature[:, 1] / np.amax(np.abs(emb_feature[:, 1]))
    print('TSNE done.')

    ## architecture density
    fig, ax = plt.subplots(figsize=(5, 5))
    xedges = np.linspace(-1, 1.04, 52)
    yedges = np.linspace(-1, 1.04, 52)
    H, xedges, yedges, img = ax.hist2d(emb_x, emb_y, bins=(xedges, yedges))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax, ax=ax)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.set_label('Density')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'density-{}-{} points.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.close()


    # accuracy distribution
    xw = xedges[1] - xedges[0]
    yw = yedges[1] - yedges[0]
    x_cor = np.floor((emb_x - xedges[0]) / xw).astype(int)
    y_cor = np.floor((emb_y - yedges[0]) / yw).astype(int)
    acc = np.zeros((51, 51))
    for xx in range(51):
        for yy in range(51):
            idx = np.logical_and((x_cor == xx), (y_cor == yy))
            if idx.any():
                acc[xx, yy] = np.mean(test_acc[idx])
    xx = (np.linspace(0, 50, 51) + 0.5) * xw + xedges[0]
    yy = (np.linspace(0, 50, 51) + 0.5) * yw + yedges[0]

    ma_acc = np.ma.masked_where(acc == 0, acc)
    palette = copy(plt.cm.viridis)
    palette.set_over('r', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('w', 1.0)

    ## raw version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_acc.T,
                   cmap=palette,
                   norm=colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Test Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_raw.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.close()

    ## smooth version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_acc.T,
                   cmap=palette,
                   interpolation='bilinear',
                   norm=colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Test Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_smooth.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.close()

    ## scatter version
    x1 = emb_x[test_acc >= 0.94]
    y1 = emb_y[test_acc >= 0.94]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.scatter(emb_x, emb_y, c=test_acc, s=1, cmap=palette, norm=colors.Normalize(vmin=0.84, vmax=0.94))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax.scatter(x1, y1, c='r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Test Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_scatter.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.close()

def visualize2D_supverised(emb_path, sup_emb_path, out_path, seed=0, sample_num=1000, acc_dyn=(0.82, 0.92)):
    ## load embedding
    dataset = torch.load(emb_path)
    dataset_sup = np.squeeze(np.load(sup_emb_path))
    _, emb_name = os.path.split(emb_path)
    feature = []
    test_acc = []
    if sample_num <= 0 or sample_num >= len(dataset):
        sample_idx = range(len(dataset))
        sample_num = len(dataset)
    else:
        sample_idx = sample_without_replacement(len(dataset), sample_num, random_state=0)
    for j in tqdm(range(len(sample_idx)), desc='load feature'):
        i = sample_idx[j]
        feature.append(dataset_sup[i])
        test_acc.append(dataset[i]['test_accuracy'])
    feature = np.stack(feature, axis=0)
    test_acc = np.stack(test_acc, axis=0)
    ## tsne reduces dim
    print('TSNE...')
    tsne = TSNE(random_state=seed)
    emb_feature = tsne.fit_transform(feature)
    emb_x = emb_feature[:, 0] / np.amax(np.abs(emb_feature[:, 0]))
    emb_y = emb_feature[:, 1] / np.amax(np.abs(emb_feature[:, 1]))
    print('TSNE done.')

    ## architecture density
    fig, ax = plt.subplots(figsize=(5, 5))
    xedges = np.linspace(-1, 1.04, 52)
    yedges = np.linspace(-1, 1.04, 52)
    H, xedges, yedges, img = ax.hist2d(emb_x, emb_y, bins=(xedges, yedges))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax, ax=ax)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.set_label('Density')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'density-{}-{} points.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.show()
    plt.close()


    # accuracy distribution
    xw = xedges[1] - xedges[0]
    yw = yedges[1] - yedges[0]
    x_cor = np.floor((emb_x - xedges[0]) / xw).astype(int)
    y_cor = np.floor((emb_y - yedges[0]) / yw).astype(int)
    acc = np.zeros((51, 51))
    for xx in range(51):
        for yy in range(51):
            idx = np.logical_and((x_cor == xx), (y_cor == yy))
            if idx.any():
                acc[xx, yy] = np.mean(test_acc[idx])
    xx = (np.linspace(0, 50, 51) + 0.5) * xw + xedges[0]
    yy = (np.linspace(0, 50, 51) + 0.5) * yw + yedges[0]

    ma_acc = np.ma.masked_where(acc == 0, acc)
    palette = copy(plt.cm.viridis)
    palette.set_over('r', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('w', 1.0)

    ## raw version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_acc.T,
                   cmap=palette,
                   norm=colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Test Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_raw.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.show()
    plt.close()

    ## smooth version
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_acc.T,
                   cmap=palette,
                   interpolation='bilinear',
                   norm=colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
                   origin='lower',
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Test Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_smooth.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.show()
    plt.close()

    ## scatter version
    x1 = emb_x[test_acc >= 0.94]
    y1 = emb_y[test_acc >= 0.94]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.scatter(emb_x, emb_y, c=test_acc, s=1, cmap=palette, norm=colors.Normalize(vmin=0.84, vmax=0.94))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax.scatter(x1, y1, c='r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both')
    cbar.set_label('Test Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_scatter.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.show()
    plt.close()

def visualize2D_supverised_nas201(acc_path, sup_emb_path, out_path, seed=0, sample_num=1000, acc_dyn=(0.82, 0.92)):
    ## load embedding
    emb_name = 'cfar10'
    dataset = np.load(sup_emb_path)
    dataset_acc = np.load(acc_path)


    feature = []
    test_acc = []
    sample_idx = sample_without_replacement(len(dataset), sample_num, random_state=0)
    for j in tqdm(range(len(sample_idx)), desc='load feature'):
        i = sample_idx[j]
        feature.append(dataset[i])
        test_acc.append(dataset_acc[i])
    feature = np.stack(feature, axis=0)
    test_acc = np.stack(test_acc, axis=0) / 100.

    ## tsne reduces dim
    print('TSNE...')
    tsne = TSNE(random_state=seed, early_exaggeration=30)
    emb_feature = tsne.fit_transform(feature)
    emb_x = emb_feature[:, 0] / np.amax(np.abs(emb_feature[:, 0]))
    emb_y = emb_feature[:, 1] / np.amax(np.abs(emb_feature[:, 1]))
    print('TSNE done.')

    xedges = np.linspace(-1, 1.04, 52)
    yedges = np.linspace(-1, 1.04, 52)


    # accuracy distribution
    xw = xedges[1] - xedges[0]
    yw = yedges[1] - yedges[0]
    x_cor = np.floor((emb_x - xedges[0]) / xw).astype(int)
    y_cor = np.floor((emb_y - yedges[0]) / yw).astype(int)
    acc = np.zeros((51, 51))
    for xx in range(51):
        for yy in range(51):
            idx = np.logical_and((x_cor == xx), (y_cor == yy))
            if idx.any():
                acc[xx, yy] = np.mean(test_acc[idx])
    xx = (np.linspace(0, 50, 51) + 0.5) * xw + xedges[0]
    yy = (np.linspace(0, 50, 51) + 0.5) * yw + yedges[0]

    ma_acc = np.ma.masked_where(acc == 0, acc)

    palette = copy(plt.cm.viridis)
    palette.set_over('r', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('w', 1.0)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_acc.T,
                   cmap=palette,
                   origin='lower',
                   norm=colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    # norm = colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both') #, extend='both'
    cbar.set_label('Test Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_raw.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.show()
    plt.close()


def visualize2D_unsupverised_nas201(acc_path, emb_path, out_path, seed=0, sample_num=1000, acc_dyn=(0.82, 0.92)):
    ## load embedding
    emb_name = 'nas201'
    dataset = torch.load(emb_path)
    dataset_acc = np.load(acc_path)


    feature = []
    test_acc = []
    sample_idx = sample_without_replacement(len(dataset), sample_num, random_state=0)
    for j in tqdm(range(len(sample_idx)), desc='load feature'):
        i = sample_idx[j]
        feature.append(dataset[i]['feature'].detach().numpy())
        test_acc.append(dataset_acc[i])
    feature = np.stack(feature, axis=0)
    test_acc = np.stack(test_acc, axis=0) / 100.

    ## tsne reduces dim
    print('TSNE...')
    tsne = TSNE(random_state=seed, early_exaggeration=30)
    emb_feature = tsne.fit_transform(feature)
    emb_x = emb_feature[:, 0] / np.amax(np.abs(emb_feature[:, 0]))
    emb_y = emb_feature[:, 1] / np.amax(np.abs(emb_feature[:, 1]))
    print('TSNE done.')

    xedges = np.linspace(-1, 1.04, 52)
    yedges = np.linspace(-1, 1.04, 52)


    # accuracy distribution
    xw = xedges[1] - xedges[0]
    yw = yedges[1] - yedges[0]
    x_cor = np.floor((emb_x - xedges[0]) / xw).astype(int)
    y_cor = np.floor((emb_y - yedges[0]) / yw).astype(int)
    acc = np.zeros((51, 51))
    for xx in range(51):
        for yy in range(51):
            idx = np.logical_and((x_cor == xx), (y_cor == yy))
            if idx.any():
                acc[xx, yy] = np.mean(test_acc[idx])
    xx = (np.linspace(0, 50, 51) + 0.5) * xw + xedges[0]
    yy = (np.linspace(0, 50, 51) + 0.5) * yw + yedges[0]

    ma_acc = np.ma.masked_where(acc == 0, acc)

    palette = copy(plt.cm.viridis)
    palette.set_over('r', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('w', 1.0)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    im = ax.imshow(ma_acc.T,
                   cmap=palette,
                   origin='lower',
                   norm=colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
                   extent=[xx[0], xx[-1], yy[0], yy[-1]])
    # norm = colors.Normalize(vmin=acc_dyn[0], vmax=acc_dyn[1]),
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ax=ax, extend='both') #, extend='both'
    cbar.set_label('Test Accuracy')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(out_path, 'accuracy-{}-{} points_raw.png'.format(emb_name, sample_num)),
                bbox_inches='tight')
    plt.show()
    plt.close()

def distribution(emb_path, sup_emb_path, sample_num, seed):
    ## load embedding
    dataset = torch.load(emb_path)
    _, emb_name = os.path.split(emb_path)
    feature = []
    test_acc = []
    sample_idx = sample_without_replacement(len(dataset), sample_num, random_state=0)
    for j in tqdm(range(len(sample_idx)), desc='load feature'):
        i = sample_idx[j]
        feature.append(dataset[i]['feature'].detach().numpy())
    feature = np.stack(feature, axis=0)
    ## tsne reduces dim
    print('TSNE...')
    tsne = TSNE(random_state=seed)
    emb_feature = tsne.fit_transform(feature)
    emb_x = emb_feature[:, 0] / np.amax(np.abs(emb_feature[:, 0]))
    emb_y = emb_feature[:, 1] / np.amax(np.abs(emb_feature[:, 1]))
    print('TSNE done.')

    ## load supervised embedding
    sup_dataset = np.squeeze(np.load(sup_emb_path))
    sup_feature = []
    test_acc = []
    sample_idx = sample_without_replacement(len(sup_dataset), sample_num, random_state=0)
    for j in tqdm(range(len(sample_idx)), desc='load feature'):
        i = sample_idx[j]
        sup_feature.append(sup_dataset[i])
    sup_feature = np.stack(sup_feature, axis=0)
    ## tsne reduces dim
    print('TSNE...')
    tsne = TSNE(random_state=seed)
    sup_emb_feature = tsne.fit_transform(sup_feature)
    sup_emb_x = sup_emb_feature[:, 0] / np.amax(np.abs(sup_emb_feature[:, 0]))
    sup_emb_y = sup_emb_feature[:, 1] / np.amax(np.abs(sup_emb_feature[:, 1]))
    print('TSNE done.')

    ## 2d histogram
    palette = copy(plt.cm.viridis)
    palette.set_bad('w', 1.0)
    palette.set_over('y', 1.0)
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 3), ncols=2, nrows=1)
    xedges = np.linspace(-1, 1.04, 52)
    yedges = np.linspace(-1, 1.04, 52)
    H1, xedges, yedges = np.histogram2d(emb_x, emb_y, bins=(xedges, yedges))
    H1 = H1.T
    H1 = np.ma.masked_where(H1 == 0, H1)

    xedges = np.linspace(-1, 1.04, 52)
    yedges = np.linspace(-1, 1.04, 52)
    H2, xedges, yedges = np.histogram2d(sup_emb_x, sup_emb_y, bins=(xedges, yedges))
    H2 = H2.T
    H2 = np.ma.masked_where(H2 == 0, H2)
    X, Y = np.meshgrid(xedges, yedges)

    ax1.pcolormesh(X, Y, H1 / np.max([np.max(H1), np.max(H2)]), cmap=palette, norm=colors.Normalize(vmin=0, vmax=1))
    im = ax2.pcolormesh(X, Y, H2 / np.max([np.max(H1), np.max(H2)]), cmap=palette, norm=colors.Normalize(vmin=0, vmax=1))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = plt.colorbar(im, cax=cax, ax=ax2)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    cbar.set_label('Density')
    plt.tight_layout()
    plt.savefig('density_compare.pdf', bbox_inches='tight')
    plt.show()





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Embedding 2-dimensional Plot')
    parser.add_argument('--emb_path', type=str, metavar='PATH',
                        default=None, help='unsupervised embedding file (default: None)')
    parser.add_argument('--supervised_emb_path', type=str, metavar='PATH',
                        default=None, help='supervised embedding file (default: None)')
    parser.add_argument('--output_path', type=str, default=None, metavar='PATH', help='output path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_nums', type=int, default=10000, metavar='N',
                        help='sample points, -1 means all (default: 10000)')
    parser.add_argument('--acc_dyn', type=tuple, default=(0.82, 0.92), metavar='TUPLE',
                        help='major dynamics of test accuracy (default: (0.82, 0.92))')
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    visualize2D(args.emb_path, args.output_path, args.seed, args.sample_nums, args.acc_dyn)