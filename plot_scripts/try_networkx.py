import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def node_match(n1, n2):
    if n1['op'] == n2['op']:
        return True
    else:
        return False

def edge_match(e1, e2):
    return True

def gen_graph(adj, ops):
    G = nx.DiGraph()
    for k, op in enumerate(ops):
        G.add_node(k, op=op)
    assert adj.shape[0] == adj.shape[1] == len(ops)
    for row in range(len(ops)):
        for col in range(row + 1, len(ops)):
            if adj[row, col] > 0:
                G.add_edge(row, col)
    return G

def preprocess_adj_op(adj, op):
    def counting_trailing_false(l):
        count = 0
        for TF in l[-1::-1]:
            if TF:
                break
            else:
                count += 1
        return count

    def transform_op(op):
        idx2op = {0:'input', 1:'conv1x1-bn-relu', 2:'conv3x3-bn-relu', 3:'maxpool3x3', 4:'output'}
        return [idx2op[idx] for idx in op.argmax(axis=1)]

    adj = np.array(adj).astype(int)
    op = np.array(op).astype(int)

    assert op.shape[0] == adj.shape[0] == adj.shape[1]
    # find all zero columns
    adj_zero_col = counting_trailing_false(adj.any(axis=0))
    # find all zero rows
    adj_zero_row = counting_trailing_false(adj.any(axis=1))
    # find all zero rows
    op_zero_row = counting_trailing_false(op.any(axis=1))
    assert adj_zero_col == op_zero_row == adj_zero_row - 1, 'Inconsistant result {}={}={}'.format(adj_zero_col, op_zero_row, adj_zero_row - 1)
    N = op.shape[0] - adj_zero_col
    adj = adj[:N, :N]
    op = op[:N]

    return adj, transform_op(op)



if __name__ == '__main__':

    adj1 = np.array([[0, 1, 1, 1, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0]])
    op1 = ['in', 'conv1x1', 'conv3x3', 'mp3x3', 'out']

    adj2 = np.array([[0, 1, 1, 1, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0]])
    op2 = ['in', 'conv1x1', 'mp3x3', 'conv3x3', 'out']


    adj3 = np.array([[0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0]])
    op3 = ['in', 'conv1x1', 'conv3x3', 'mp3x3', 'out','out2']

    adj4 = np.array([[0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    op4 = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]])
    adj4, op4 = preprocess_adj_op(adj4, op4)



    G1 = gen_graph(adj1, op1)
    G2 = gen_graph(adj2, op2)
    G3 = gen_graph(adj3, op3)
    G4 = gen_graph(adj4, op4)


    plt.subplot(141)
    nx.draw(G1, with_labels=True, font_weight='bold')
    plt.subplot(142)
    nx.draw(G2, with_labels=True, font_weight='bold')
    plt.subplot(143)
    nx.draw(G3, with_labels=True, font_weight='bold')
    plt.subplot(144)
    nx.draw(G4, with_labels=True, font_weight='bold')

    nx.graph_edit_distance(G1,G2, node_match=node_match, edge_match=edge_match)
    nx.graph_edit_distance(G2,G3, node_match=node_match, edge_match=edge_match)