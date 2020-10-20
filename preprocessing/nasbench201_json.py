"""API source: https://github.com/D-X-Y/NAS-Bench-201/blob/v1.1/nas_201_api/api.py"""
from api import NASBench201API as API
import numpy as np
import json
from collections import OrderedDict

nas_bench = API('data/NAS-Bench-201-v1_0-e61699.pth')



# num = len(api)
# for i, arch_str in enumerate(api):
#   print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))
#
# info = api.query_meta_info_by_index(1)  # This is an instance of `ArchResults`
# res_metrics = info.get_metrics('cifar10', 'train') # This is a dict with metric names as keys
# cost_metrics = info.get_comput_costs('cifar100') # This is a dict with metric names as keys, e.g., flops, params, latency
#
# api.show(1)
# api.show(2)

def info2mat(arch_index):
    #info.all_results

    info = nas_bench.query_meta_info_by_index(arch_index)
    ops = {'input':0, 'nor_conv_1x1':1, 'nor_conv_3x3':2, 'avg_pool_3x3':3, 'skip_connect':4, 'none':5, 'output':6}
    adj_mat = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1 ,0 ,0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0]])

    nodes = ['input']
    steps = info.arch_str.split('+')
    steps_coding = ['0', '0', '1', '0', '1', '2']
    cont = 0
    for step in steps:
        step = step.strip('|').split('|')
        for node in step:
            n, idx = node.split('~')
            assert idx == steps_coding[cont]
            cont += 1
            nodes.append(n)
    nodes.append('output')

    node_mat =np.zeros([8, len(ops)]).astype(int)
    ops_idx = [ops[k] for k in nodes]
    node_mat[[0,1,2,3,4,5,6,7],ops_idx] = 1

    # For cifar10-valid with converged
    valid_acc, val_acc_avg, time_cost, test_acc, test_acc_avg = train_and_eval(arch_index, nepoch=None, dataname='cifar10-valid', use_converged_LR=True)
    cifar10_valid_converged = { 'test_accuracy': test_acc,
                                'test_accuracy_avg': test_acc_avg,
                                'validation_accuracy':valid_acc,
                                'validation_accuracy_avg': val_acc_avg,
                                'module_adjacency':adj_mat.tolist(),
                                'module_operations': node_mat.tolist(),
                                'training_time': time_cost}


    # For cifar100
    valid_acc, val_acc_avg, time_cost, test_acc, test_acc_avg = train_and_eval(arch_index, nepoch=199, dataname='cifar100', use_converged_LR=False)
    cifar100                = {'test_accuracy': test_acc,
                               'test_accuracy_avg': test_acc_avg,
                               'validation_accuracy': valid_acc,
                               'validation_accuracy_avg': val_acc_avg,
                               'module_adjacency': adj_mat.tolist(),
                               'module_operations': node_mat.tolist(),
                               'training_time': time_cost}

    # For ImageNet16-120
    valid_acc, val_acc_avg, time_cost, test_acc, test_acc_avg = train_and_eval(arch_index, nepoch=199, dataname='ImageNet16-120', use_converged_LR=False)
    ImageNet16_120          = {'test_accuracy': test_acc,
                               'test_accuracy_avg': test_acc_avg,
                               'validation_accuracy': valid_acc,
                               'validation_accuracy_avg': val_acc_avg,
                               'module_adjacency': adj_mat.tolist(),
                               'module_operations': node_mat.tolist(),
                               'training_time': time_cost}


    return {'cifar10_valid_converged': cifar10_valid_converged,
            'cifar100':cifar100,
            'ImageNet16_120': ImageNet16_120 }

def train_and_eval(arch_index, nepoch=None, dataname=None, use_converged_LR=True):
    assert dataname !='cifar10', 'Do not allow cifar10 dataset'
    if use_converged_LR and dataname=='cifar10-valid':
        assert nepoch == None, 'When using use_converged_LR=True, please set nepoch=None, use 12-converged-epoch by default.'


        info = nas_bench.get_more_info(arch_index, dataname, None, True)
        valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
        valid_acc_avg = nas_bench.get_more_info(arch_index, 'cifar10-valid', None, False, False)['valid-accuracy']
        test_acc = nas_bench.get_more_info(arch_index, 'cifar10', None, False, True)['test-accuracy']
        test_acc_avg = nas_bench.get_more_info(arch_index, 'cifar10', None, False, False)['test-accuracy']

    elif not use_converged_LR:

        assert isinstance(nepoch, int), 'nepoch should be int'
        xoinfo = nas_bench.get_more_info(arch_index, 'cifar10-valid', None, True)
        xocost = nas_bench.get_cost_info(arch_index, 'cifar10-valid', False)
        info = nas_bench.get_more_info(arch_index, dataname, nepoch, False, True)
        cost = nas_bench.get_cost_info(arch_index, dataname, False)
        # The following codes are used to estimate the time cost.
        # When we build NAS-Bench-201, architectures are trained on different machines and we can not use that time record.
        # When we create checkpoints for converged_LR, we run all experiments on 1080Ti, and thus the time for each architecture can be fairly compared.
        nums = {'ImageNet16-120-train': 151700, 'ImageNet16-120-valid': 3000,
                'cifar10-valid-train' : 25000,  'cifar10-valid-valid' : 25000,
                'cifar100-train'      : 50000,  'cifar100-valid'      : 5000}
        estimated_train_cost = xoinfo['train-per-time'] / nums['cifar10-valid-train'] * nums['{:}-train'.format(dataname)] / xocost['latency'] * cost['latency'] * nepoch
        estimated_valid_cost = xoinfo['valid-per-time'] / nums['cifar10-valid-valid'] * nums['{:}-valid'.format(dataname)] / xocost['latency'] * cost['latency']
        try:
            valid_acc, time_cost = info['valid-accuracy'], estimated_train_cost + estimated_valid_cost
        except:
            valid_acc, time_cost = info['est-valid-accuracy'], estimated_train_cost + estimated_valid_cost
        test_acc = info['test-accuracy']
        test_acc_avg = nas_bench.get_more_info(arch_index, dataname, None, False, False)['test-accuracy']
        valid_acc_avg = nas_bench.get_more_info(arch_index, dataname, None, False, False)['valid-accuracy']
    else:
        # train a model from scratch.
        raise ValueError('NOT IMPLEMENT YET')
    return valid_acc, valid_acc_avg, time_cost, test_acc, test_acc_avg


def enumerate_dataset(dataset):
    for k in range(len(nas_bench)):
        print('{}: {}/{}'.format(dataset, k,len(nas_bench)))
        res = info2mat(k)
        yield {k:res[dataset]}

def gen_json_file(dataset):
    data_dict = OrderedDict()
    enum_dataset = enumerate_dataset(dataset)
    for data_point in enum_dataset:
        data_dict.update(data_point)
    with open('data/{}.json'.format(dataset), 'w') as outfile:
        json.dump(data_dict, outfile)

if __name__=='__main__':

    for dataset in ['cifar10_valid_converged', 'cifar100', 'ImageNet16_120']:
        gen_json_file(dataset)
