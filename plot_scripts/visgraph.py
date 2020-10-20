import argparse
import numpy as np
import igraph
import pygraphviz as pgv
import torch
from tqdm import tqdm
from numpy import linalg
import numpy.ma as ma
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
sys.path.insert(0, os.getcwd())
import darts.cnn.genotypes
from plot_scripts import draw_darts
from utils.utils import load_json

# ### example of plat DAG
# adj = torch.tensor([[0, 1, 0, 0, 0, 0, 1],
#                     [0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 1],
#                     [0, 0, 0, 0, 1, 1, 0],
#                     [0, 0, 0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 0, 0, 1],
#                     [0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
#
# ops = torch.tensor([[1, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 0, 1, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 0, 0, 1, 0],
#                     [0, 0, 0, 0, 1]], dtype=torch.int32)
#
# G = adj2graph(ops, adj)
# file_name = plot_DAG(G[0], os.path.curdir, 'example')

# define a adjacent matrix of straight networks
s0_adj = torch.LongTensor([[0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0]])

def read_feature(emb_path):
    dataset = torch.load(emb_path)
    feature = []
    test_acc = []
    for i in tqdm(range(len(dataset)), desc='load feature'):
        feature.append(dataset[i]['feature'].detach().numpy())
        test_acc.append(dataset[i]['test_accuracy'])
    feature = np.stack(feature, axis=0)
    test_acc = np.stack(test_acc, axis=0)
    return feature, test_acc
    # _, emb_name = os.path.split(emb_path)
    # if os.path.exists(os.path.join('feature', 'dim-{}'.format(emb_dim), emb_name)) and not rewrite:
    #     return os.path.join('feature', 'dim-{}'.format(emb_dim), emb_name)
    # else:
    #     if not os.path.exists('feature'):
    #         os.makedirs('feature')
    #     if not os.path.exists(os.path.join('feature', 'dim-{}'.format(emb_dim))):
    #         os.makedirs(os.path.join('feature', 'dim-{}'.format(emb_dim)))
    #     embedding = torch.load(emb_path)
    #     feature = torch.zeros(len(embedding), emb_dim, dtype=torch.float32)
    #     for i in range(len(embedding)):
    #         feature[i] = embedding[i]['feature'].detach()
    #     torch.save(feature, os.path.join('feature', 'dim-{}'.format(emb_dim), emb_name))
    #     return os.path.join('feature', 'dim-{}'.format(emb_dim), emb_name)


def adj2graph(ops, adj):
    if ops.dim() == 2:
        ops = ops.unsqueeze(0)
        adj = adj.unsqueeze(0)
    batch_size, _, _ = ops.shape
    node_ops = torch.argmax(ops, dim=2).numpy()
    ops_value = torch.max(ops, dim=2).values.numpy()
    node_num = []
    node_ops_nonzero = []
    ## delete zero operation for nasbench 101
    for i, (op, val) in enumerate(zip(node_ops, ops_value)):
        node_ops_nonzero.append(op[val == 1].tolist())
        node_num.append(np.sum(val).item())
    adj = torch.triu(adj, diagonal=1)
    G = [igraph.Graph(node_num[i],
                      torch.nonzero(adj[i]).tolist(),
                      vertex_attrs={'operation': node_ops_nonzero[i]}, directed=True)
         for i in range(batch_size)]
    return G


'''Network visualization'''
def plot_DAG(g, res_dir, name, data_type, backbone=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(res_dir, name+'.png')
    draw_network(g, file_name, data_type, backbone)
    return file_name


def draw_network(g, path, data_type, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_node(graph, idx, g.vs[idx]['operation'], data_type)
    straight_edges = []
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx - 1 and backbone:
                graph.add_edge(node, idx, weight=1)
                straight_edges.append((node, idx))
            else:
                graph.add_edge(node, idx, weight=0)
    all_straight_edges = [(i, i + 1) for i in range(g.vcount() - 1)]
    diff_straight = list(set(all_straight_edges) - set(straight_edges))
    if diff_straight:
        for e in diff_straight:
            graph.add_edge(e[0], e[1], color='white') ## white edges doesn't appear in graph, which controls shape
    graph.layout(prog='dot')
    graph.draw(path)


def add_node(graph, node_id, label, data_type='nasbench101', shape='box', style='filled'):
    if data_type == 'nasbench101':
        if label == 0:
            label = 'in'
            color = 'skyblue'
        elif label == 1:
            label = '1x1'
            color = 'pink'
        elif label == 2:
            label = '3x3'
            color = 'yellow'
        elif label == 3:
            label = 'MP'
            color = 'orange'
        elif label == 4:
            label = 'out'
            color = 'beige'
    elif data_type == 'nasbench201':
        if label == 0:
            label = 'in'
            color = 'skyblue'
        elif label == 1:
            label = '1x1'
            color = 'pink'
        elif label == 2:
            label = '3x3'
            color = 'yellow'
        elif label == 3:
            label = 'pool'
            color = 'orange'
        elif label == 4:
            label = 'skip'
            color = 'greenyellow'
        elif label == 5:
            label = 'none'
            color = 'seagreen3'
        elif label == 6:
            label = 'out'
            color = 'beige'
    elif data_type == 'darts':
        pass
    else:
        print('do not support!')
        exit()
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)

def get_straight(dataset, num=1):
    # find a straight network
    idx = []
    for i in tqdm(range(len(dataset)), desc='find {} straight nets'.format(num)):
        tmp = torch.LongTensor(dataset[str(i)]['module_adjacency'])
        if torch.all(s0_adj == tmp):
            idx.append(i)
    idx = np.stack(idx)
    if num == 1:
        return [np.random.choice(idx, num).tolist()]
    else:
        return np.random.choice(idx, num).tolist()


def smooth_exp(data_path, emb_path, supervised_emb_path, output_path, data_type, random_path, path_step, straight_path):
    print('experiments:')
    ## load raw architecture
    dataset = load_json(data_path)
    ## load feature & test_acc
    feature, test_acc = read_feature(emb_path)
    feature_sup = np.squeeze(np.load(supervised_emb_path))
    feature_nums = len(dataset)
    ## get start points
    start_idx = np.random.choice(feature_nums, random_path, replace=False).tolist()
    if straight_path > 0:
        straight_idx = get_straight(dataset, num=straight_path)
        start_idx = np.stack(start_idx + straight_idx)
    ## smooth experiments
    ops = []
    adj = []
    ops_sup = []
    adj_sup = []
    for k, ind in enumerate(start_idx):
        ops_k = []
        adj_k = []
        prev_node = feature[ind].reshape(1, -1)
        mask = np.zeros(feature_nums, dtype=int)
        ## supervised
        ops_k_sup = []
        adj_k_sup = []
        prev_node_sup = feature_sup[ind].reshape(1, -1)
        mask_sup = np.zeros(feature_nums, dtype=int)
        for i in tqdm(range(path_step), desc='smooth experiment {} of {}'.format(k+1, len(start_idx))):
            dis = linalg.norm(feature - prev_node, axis=1)
            mdis = ma.masked_array(dis, mask)
            idx = np.argmin(mdis)
            mask[idx] = 1
            prev_node = feature[idx].reshape(1, -1)
            ops_k.append(torch.LongTensor(dataset[str(idx)]['module_operations']))
            adj_k.append(torch.LongTensor(dataset[str(idx)]['module_adjacency']))
            ## supervised
            dis_sup = linalg.norm(feature_sup - prev_node_sup, axis=1)
            mdis_sup = ma.masked_array(dis_sup, mask_sup)
            idx_sup = np.argmin(mdis_sup)
            mask_sup[idx_sup] = 1
            prev_node_sup = feature_sup[idx_sup].reshape(1, -1)
            ops_k_sup.append(torch.LongTensor(dataset[str(idx_sup)]['module_operations']))
            adj_k_sup.append(torch.LongTensor(dataset[str(idx_sup)]['module_adjacency']))
        ops_k = torch.stack(ops_k)
        adj_k = torch.stack(adj_k)
        ops.append(ops_k)
        adj.append(adj_k)
        ops_k_sup = torch.stack(ops_k_sup)
        adj_k_sup = torch.stack(adj_k_sup)
        ops_sup.append(ops_k_sup)
        adj_sup.append(adj_k_sup)

    ## conver to graph
    for i in tqdm(range(len(start_idx)), desc='draw graphs'):
        G = adj2graph(ops[i], adj[i])
        names = []
        temp_path = '.temp'
        G_sup = adj2graph(ops_sup[i], adj_sup[i])
        names_sup = []
        temp_path_sup = '.temp_sup'
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        if not os.path.exists(temp_path_sup):
            os.makedirs(temp_path_sup)
        for j in range(path_step):
            namej = plot_DAG(G[j], temp_path, str(j), data_type, backbone=True)
            names.append(namej)
            namej_sup = plot_DAG(G_sup[j], temp_path_sup, str(j), data_type, backbone=True)
            names_sup.append(namej_sup)
        ## pave to single image
        if not os.path.exists(os.path.join(output_path, 'unsupervised')):
            os.makedirs(os.path.join(output_path, 'unsupervised'))
        images = [[Image.open(name) for name in names]]
        join_images(
            *images,
            bg_color='white',
            alignment=(0, 0)
        ).save(os.path.join(output_path, 'unsupervised', '{}_unsupervised.png'.format(start_idx[i])))

        if not os.path.exists(os.path.join(output_path, 'supervised')):
            os.makedirs(os.path.join(output_path, 'supervised'))
        images = [[Image.open(name) for name in names_sup]]
        join_images(
            *images,
            bg_color='white',
            alignment=(0, 0)
        ).save(os.path.join(output_path, 'supervised', '{}_supervised.png'.format(start_idx[i])))

        if not os.path.exists(os.path.join(output_path, 'compare')):
            os.makedirs(os.path.join(output_path, 'compare'))
        images = [[Image.open(name) for name in names], [Image.open(name) for name in names_sup]]
        join_images(
            *images,
            bg_color='white',
            alignment=(0, 0)
        ).save(os.path.join(output_path, 'compare', '{}_compare.png'.format(start_idx[i])))

def join_images(*rows, bg_color=(0, 0, 0, 0), alignment=(0.5, 0.5)):
    rows = [
        [image.convert('RGBA') for image in row]
        for row
        in rows
    ]

    heights = [
        max(image.height for image in row)
        for row
        in rows
    ]

    widths = [
        max(image.width for image in column)
        for column
        in zip(*rows)
    ]

    tmp = Image.new(
        'RGBA',
        size=(sum(widths), sum(heights)),
        color=bg_color
    )

    for i, row in enumerate(rows):
        for j, image in enumerate(row):
            y = sum(heights[:i]) + int((heights[i] - image.height) * alignment[1])
            x = sum(widths[:j]) + int((widths[j] - image.width) * alignment[0])
            tmp.paste(image, (x, y))

    return tmp


def join_images_horizontally(*row, bg_color=(0, 0, 0), alignment=(0.5, 0.5)):
    return join_images(
        row,
        bg_color=bg_color,
        alignment=alignment
    )


def join_images_vertically(*column, bg_color=(0, 0, 0), alignment=(0.5, 0.5)):
    return join_images(
        *[[image] for image in column],
        bg_color=bg_color,
        alignment=alignment
    )

def read_feature_darts(emb_path):
    dataset = torch.load(emb_path)
    feature = []
    geno = []
    for i in tqdm(range(len(dataset)), desc='load feature and genotype'):
        feature.append(dataset[i]['feature'].detach().numpy())
        geno.append(dataset[i]['genotype'])
    feature = np.stack(feature, axis=0)
    return feature, geno

def smooth_exp_darts(emb_path, output_path, random_path, path_step):
    print('experiments (DARTS):')
    ## load feature & genotype
    feature, geno = read_feature_darts(emb_path)
    feature_nums = len(geno)
    ## get start points
    start_idx = np.random.choice(feature_nums, random_path, replace=False).tolist()
    ## smooth experiments
    nets = []
    for k, idx in enumerate(start_idx):
        net_k = []
        prev_node = feature[idx].reshape(1, -1)
        mask = np.zeros(feature_nums, dtype=int)
        for i in tqdm(range(path_step), desc='smooth experiment {} of {}'.format(k+1, len(start_idx))):
            dis = linalg.norm(feature - prev_node, axis=1)
            mdis = ma.masked_array(dis, mask)
            idx = np.argmin(mdis)
            mask[idx] = 1
            prev_node = feature[idx].reshape(1, -1)
            net_k.append(geno[idx])
        nets.append(net_k)
    ## draw graphs
    for i in tqdm(range(len(start_idx)), desc='draw graphs'):
        names = []
        temp_path = '.temp'
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        for j in range(path_step):
            namej = os.path.join(temp_path, str(j))
            draw_darts.plot(nets[i][j], namej)
            namej = os.path.join(temp_path, '{}.png'.format(j))
            names.append(namej)
        ## pave to single image
        images = [[Image.open(name)] for name in names]
        join_images(
            *images,
            bg_color='white',
            alignment=(0, 0)
        ).save(os.path.join(output_path, '{}.png'.format(start_idx[i])))

def smooth_exp_nas201(data_path, emb_path, supervised_emb_path, output_path, random_path, path_step):
    print('experiments (NAS 201):')
    ## load raw architecture
    dataset = load_json(data_path)
    ## load feature & test_acc
    feature_raw = torch.load(emb_path)
    feature = []
    for i in tqdm(range(len(feature_raw)), desc='load feature'):
        feature.append(feature_raw[i]['feature'].detach().numpy())
    feature = np.stack(feature)
    feature_sup = np.load(supervised_emb_path)
    feature_nums = len(dataset)
    ## get start points
    start_idx = np.random.choice(feature_nums, random_path, replace=False).tolist()
    ## smooth experiments
    ops = []
    ops_sup = []

    for k, ind in enumerate(start_idx):
        ops_k = []
        prev_node = feature[ind].reshape(1, -1)
        mask = np.zeros(feature_nums, dtype=int)
        ## supervised
        ops_k_sup = []
        prev_node_sup = feature_sup[ind].reshape(1, -1)
        mask_sup = np.zeros(feature_nums, dtype=int)
        for i in tqdm(range(path_step), desc='smooth experiment {} of {}'.format(k+1, len(start_idx))):
            dis = linalg.norm(feature - prev_node, axis=1)
            mdis = ma.masked_array(dis, mask)
            idx = np.argmin(mdis)
            mask[idx] = 1
            prev_node = feature[idx].reshape(1, -1)
            ops_k.append(np.argmax(np.array(dataset[str(idx)]['module_operations']), axis=1))
            ## supervised
            dis_sup = linalg.norm(feature_sup - prev_node_sup, axis=1)
            mdis_sup = ma.masked_array(dis_sup, mask_sup)
            idx_sup = np.argmin(mdis_sup)
            mask_sup[idx_sup] = 1
            prev_node_sup = feature_sup[idx_sup].reshape(1, -1)
            ops_k_sup.append(np.argmax(np.array(dataset[str(idx_sup)]['module_operations']), axis=1))
        ops_k = np.stack(ops_k)
        ops.append(ops_k)
        ops_k_sup = np.stack(ops_k_sup)
        ops_sup.append(ops_k_sup)

    ## conver to graph
    num2ops = {0: 'in', 1: '1x1', 2: '3x3', 3: 'pool', 4: 'skip', 5: 'none', 6: 'out'}
    x = [130, 300, 280, 40, 150, 320]
    y = [550, 500, 350, 400, 250, 200]
    img = mpimg.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nas201.jpg'))
    for i in tqdm(range(len(start_idx)), desc='draw graphs'):
        names = []
        temp_path = '.temp'
        names_sup = []
        temp_path_sup = '.temp_sup'
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        if not os.path.exists(temp_path_sup):
            os.makedirs(temp_path_sup)
        ops0_prev = []
        ops1_prev = []
        for j in range(path_step):
            namej = os.path.join(temp_path, str(j)+'.jpg')
            names.append(namej)
            ops0 = [num2ops[x] for x in ops[i][j]]
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for k in range(6):
                if len(ops0_prev) == 0 or ops0[k+1] == ops0_prev[k+1]:
                    plt.text(x[k], y[k], ops0[k+1], fontsize=18, color='blue')
                else:
                    plt.text(x[k], y[k], ops0[k+1], fontsize=18, color='red')
            plt.savefig(namej, bbox_inches='tight')
            plt.close()
            ops0_prev = ops0
            namej_sup = os.path.join(temp_path_sup, str(j)+'.jpg')
            names_sup.append(namej_sup)
            ops1 = [num2ops[x] for x in ops_sup[i][j]]
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for k in range(6):
                if len(ops1_prev) == 0 or ops1[k+1] == ops1_prev[k+1]:
                    plt.text(x[k], y[k], ops1[k+1], fontsize=18, color='blue')
                else:
                    plt.text(x[k], y[k], ops1[k+1], fontsize=18, color='red')
            plt.savefig(namej_sup, bbox_inches='tight')
            plt.close()
            ops1_prev = ops1
        ## pave to single image
        if not os.path.exists(os.path.join(output_path, 'unsupervised')):
            os.makedirs(os.path.join(output_path, 'unsupervised'))
        images = [[Image.open(name) for name in names]]
        join_images(
            *images,
            bg_color='white',
            alignment=(0, 0)
        ).save(os.path.join(output_path, 'unsupervised', '{}_unsupervised.png'.format(start_idx[i])))

        if not os.path.exists(os.path.join(output_path, 'supervised')):
            os.makedirs(os.path.join(output_path, 'supervised'))
        images = [[Image.open(name) for name in names_sup]]
        join_images(
            *images,
            bg_color='white',
            alignment=(0, 0)
        ).save(os.path.join(output_path, 'supervised', '{}_supervised.png'.format(start_idx[i])))

        if not os.path.exists(os.path.join(output_path, 'compare')):
            os.makedirs(os.path.join(output_path, 'compare'))
        images = [[Image.open(name) for name in names], [Image.open(name) for name in names_sup]]
        join_images(
            *images,
            bg_color='white',
            alignment=(0, 0)
        ).save(os.path.join(output_path, 'compare', '{}_compare.png'.format(start_idx[i])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Visualize Networks (Graph)')
    parser.add_argument('--data_type', type=str, default='nasbench101', help='benchmark type (default: nasbench101)',
                        choices=['nasbench101', 'nasbench201', 'darts'], metavar='TYPE')
    parser.add_argument('--data_path', type=str, default=None, help='data *.json file (default: None)', metavar='PATH')
    parser.add_argument('--emb_path', type=str, default=None, help='unsupervised embedding *.pt (default: None)',
                        metavar='PATH')
    parser.add_argument('--supervised_emb_path', type=str, default=None,
                        help='supervised embedding *.pth (default: None)', metavar='PATH')
    parser.add_argument('--output_path', type=str, default=None, help='output path (default: None)', metavar='PATH')
    parser.add_argument('--random_path', type=int, default=50, help='num of paths to visualization (default: 50)',
                        metavar='N')
    parser.add_argument('--path_step', type=int, default=10, help='num of points of each visualization (default: 10)',
                        metavar='N')
    parser.add_argument('--straight_path', type=int, default=10,
                        help='num of paths starting at a straight networks (default: 10)', metavar='N')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.data_type == 'nasbench101':
        output_path = os.path.join(args.output_path, args.data_type, '{}steps'.format(args.path_step))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, 'compare')):
            os.makedirs(os.path.join(output_path, 'compare'))
        smooth_exp(args.data_path, args.emb_path, args.supervised_emb_path, output_path, args.data_type,
                   args.random_path, args.path_step, args.straight_path)
    elif args.data_type == 'nasbench201':
        output_path = os.path.join(args.output_path, args.data_type, '{}steps'.format(args.path_step))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, 'compare')):
            os.makedirs(os.path.join(output_path, 'compare'))
        smooth_exp_nas201(args.data_path, args.emb_path, args.supervised_emb_path, output_path,
                          args.random_path, args.path_step)
    elif args.data_type == 'darts':
        output_path = os.path.join(args.output_path, args.data_type, '{}steps'.format(args.path_step))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        smooth_exp_darts(args.emb_path, output_path, args.random_path, args.path_step)
    else:
        print('not support data type')
        exit()
