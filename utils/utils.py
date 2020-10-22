import os
import json
import torch
import torch.nn.functional as F
import random
import numpy as np

def load_json(f_name):
    """load nas-bench-101 dataset."""
    with open(f_name, 'r') as infile:
        dataset = json.loads(infile.read())
    return dataset

def save_checkpoint(model, optimizer, epoch, loss, dim, name, dropout, seed):
    """Saves a checkpoint."""
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    dir_name = 'pretrained/dim-{}'.format(dim)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_path = os.path.join(dir_name, 'model-ae-{}.pt'.format(name))
    torch.save(checkpoint, f_path)


def save_checkpoint_vae(model, optimizer, epoch, loss, dim, name, dropout, seed):
    """Saves a checkpoint."""
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    dir_name = 'pretrained/dim-{}'.format(dim)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_path = os.path.join(dir_name, 'model-{}.pt'.format(name))
    torch.save(checkpoint, f_path)

def normalize_adj(A):
    D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
    D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
    DA = stacked_spmm(D_in, A)  # swap D_in and D_out
    DAD = stacked_spmm(DA, D_out)
    return DAD

def preprocessing(A, H, method, lbd=None):
    # FixMe: Attention multiplying D or lbd are not friendly with the crossentropy loss in GAE
    assert A.dim()==3

    if method == 0:
        return A, H

    if method==1:
        # Adding global node with padding
        A = F.pad(A, (0,1), 'constant', 1.0)
        A = F.pad(A, (0,0,0,1), 'constant', 0.0)
        H = F.pad(H, (0,1,0,1), 'constant', 0.0 )
        H[:, -1, -1] = 1.0

    if method==1:
        # using A^T instead of A
        # and also adding a global node
        A = A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A) # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        return DAD, H

    elif method == 2:
        assert lbd!=None
        # using lambda*A + (1-lambda)*A^T
        A = lbd * A + (1-lbd)*A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A)  # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        def prep_reverse(DAD, H):
            AD = stacked_spmm(1.0/D_in, DAD)
            A =  stacked_spmm(AD, 1.0/D_out)
            return A.triu(1), H
        return DAD, H, prep_reverse

    elif method == 3:
        # bidirectional DAG
        assert lbd != None
        # using lambda*A + (1-lambda)*A^T
        A = lbd * A + (1 - lbd) * A.transpose(-1, -2)
        def prep_reverse(A, H):
            return 1.0/lbd*A.triu(1), H
        return A, H, prep_reverse

    elif method == 4:
        A = A + A.triu(1).transpose(-1, -2)
        def prep_reverse(A, H):
            return A.triu(1), H
        return A, H, prep_reverse


def get_accuracy(inputs, targets):
    N, I, _ = inputs[0].shape
    ops_recon, adj_recon = inputs[0], inputs[1]
    ops, adj = targets[0], targets[1]
    # post processing, assume non-symmetric
    adj_recon, adj = adj_recon.triu(1), adj.triu(1)
    correct_ops = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float().mean().item()
    mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
    mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (N*I*(I-1)/2.0-adj.sum())
    threshold = 0.5 # hard threshold
    adj_recon_thre = adj_recon > threshold
    correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item()/ (N*I*(I-1)/2.0)

    ops_correct = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float()
    adj_correct = adj_recon_thre.eq(adj.type(torch.bool)).float()
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj

def get_train_acc(inputs, targets):
    acc_train = get_accuracy(inputs, targets)
    return 'training batch: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(*acc_train)

def get_train_NN_accuracy_str(inputs, targets, decoderNN, inds):
    acc_train = get_accuracy(inputs, targets)
    acc_val = get_NN_acc(decoderNN, targets, inds)
    return 'acc_ops:{0:.4f}({4:.4f}), mean_corr_adj:{1:.4f}({5:.4f}), mean_fal_pos_adj:{2:.4f}({6:.4f}), acc_adj:{3:.4f}({7:.4f}), top-{8} index acc {9:.4f}'.format(
        *acc_train, *acc_val)

def get_NN_acc(decoderNN, targets, inds):
    ops, adj = targets[0], targets[1]
    op_recon, adj_recon, op_recon_tk, adj_recon_tk, _, ind_tk_list = decoderNN.find_NN(ops, adj, inds)
    correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj, acc = get_accuracy((op_recon, adj_recon), targets)
    pred_k = torch.tensor(ind_tk_list,dtype=torch.int)
    correct = pred_k.eq(torch.tensor(inds, dtype=torch.int).view(-1,1).expand_as(pred_k))
    topk_acc = correct.sum(dtype=torch.float) / len(inds)
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj, pred_k.shape[1], topk_acc.item()

def get_val_acc(model, cfg, X_adj, X_ops, indices):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj.cuda(), ops.cuda()
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon,_ = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave

def get_val_acc_vae(model, cfg, X_adj, X_ops, indices):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj.cuda(), ops.cuda()
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon,mu, logvar = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave

def stacked_mm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def stacked_spmm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def to_operations_darts(ops):
    transform_dict = {'c_k-2': 0, 'c_k-1': 1, 'none': 2, 'max_pool_3x3': 3, 'avg_pool_3x3': 4, 'skip_connect': 5,
                      'sep_conv_3x3': 6, 'sep_conv_5x5': 7, 'dil_conv_3x3': 8, 'dil_conv_5x5': 9, 'output': 10}

    ops_array = np.zeros([11, 11], dtype='int8')
    for row, op in enumerate(ops):
        ops_array[row, op] = 1
    return ops_array

def one_hot_darts(ops):
    transform_dict = {'c_k-2': 0, 'c_k-1': 1, 'none': 2, 'max_pool_3x3': 3, 'avg_pool_3x3': 4, 'skip_connect': 5,
                      'sep_conv_3x3': 6, 'sep_conv_5x5': 7, 'dil_conv_3x3': 8, 'dil_conv_5x5': 9, 'output': 10}

    ops_array = np.zeros([11, 11], dtype='int8')
    for row, op in enumerate(ops):
        ops_array[row, op] = 1
    return ops_array

def to_ops_darts(idx):
    transform_dict = {0:'c_k-2',1:'c_k-1',2:'none',3:'max_pool_3x3',4:'avg_pool_3x3',5:'skip_connect',6:'sep_conv_3x3',7:'sep_conv_5x5',8:'dil_conv_3x3',9:'dil_conv_5x5',10:'output'}
    ops = []
    for id in idx:
        ops.append(transform_dict[id.item()])
    return ops

def to_ops_nasbench201(idx):
    transform_dict = {0:'input',1:'nor_conv_1x1',2:'nor_conv_3x3',3:'avg_pool_3x3',4:'skip_connect',5:'none',6:'output'}
    ops = []
    for id in idx:
        ops.append(transform_dict[id.item()])
    return ops

def is_valid_nasbench201(adj, ops):
    if ops[0] != 'input' or ops[-1] != 'output':
        return False
    for i in range(2, len(ops)-1):
        if ops[i] not in ['nor_conv_1x1','nor_conv_3x3','avg_pool_3x3','skip_connect','none']:
            return False
    return True

def is_valid_darts(adj, ops):
    if ops[0] != 'c_k-2' or ops[1] != 'c_k-1' or ops[-1] != 'output':
        return False
    for i in range(2, len(ops)-1):
        if ops[i] not in ['none','max_pool_3x3','avg_pool_3x3','skip_connect','sep_conv_3x3','sep_conv_5x5','dil_conv_3x3','dil_conv_5x5']:
            return False
    adj = np.array(adj)
    #B0
    if sum(adj[:2,2]) == 0 or sum(adj[:2,3]) == 0:
        return False
    if sum(adj[4:,2]) > 0 or sum(adj[4:,3]) >0:
        return False
    #B1:
    if sum(adj[:4,4]) == 0 or sum(adj[:4,5]) == 0:
        return False
    if sum(adj[6:,4]) > 0 or sum(adj[6:,5]) > 0:
        return False
    #B2:
    if sum(adj[:6,6]) == 0 or sum(adj[:6,7]) == 0:
        return False
    if sum(adj[8:,6]) > 0 or sum(adj[8:,7]) > 0:
        return False
    #B3:
    if sum(adj[:8,8]) == 0 or sum(adj[:8,9]) == 0:
        return False
    return True






