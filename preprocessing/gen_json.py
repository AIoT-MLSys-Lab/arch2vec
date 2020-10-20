from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nasbench import api
from random import randint
import json
import numpy as np
from collections import OrderedDict

# Replace this string with the path to the downloaded nasbench.tfrecord before
# executing.
NASBENCH_TFRECORD = 'data/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

def gen_data_point(nasbench):

    i = 0
    epoch = 108

    padding = [0, 0, 0, 0, 0, 0, 0]
    best_val_acc = 0
    best_test_acc = 0

    for unique_hash in nasbench.hash_iterator():
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        print('\nIterating over {} / {} unique models in the dataset.'.format(i, 423623))
        test_acc_avg = 0.0
        val_acc_avg = 0.0
        training_time = 0.0
        for repeat_index in range(len(computed_metrics[epoch])):
            assert len(computed_metrics[epoch])==3, 'len(computed_metrics[epoch]) should be 3'
            data_point = computed_metrics[epoch][repeat_index]
            val_acc_avg += data_point['final_validation_accuracy']
            test_acc_avg += data_point['final_test_accuracy']
            training_time += data_point['final_training_time']
        val_acc_avg = val_acc_avg/3.0
        test_acc_avg = test_acc_avg/3.0
        training_time_avg = training_time/3.0
        ops_array = transform_operations(fixed_metrics['module_operations'])
        adj_array = fixed_metrics['module_adjacency'].tolist()
        model_spec = api.ModelSpec(fixed_metrics['module_adjacency'], fixed_metrics['module_operations'])
        data = nasbench.query(model_spec, epochs=108)
        print('api training time: {}'.format(data['training_time']))
        print('real training time: {}'.format(training_time_avg))

        # pad zero to adjacent matrix that has nodes less than 7
        if len(adj_array) <= 6:
            for row in range(len(adj_array)):
                for _ in range(7-len(adj_array)):
                    adj_array[row].append(0)
            for _ in range(7-len(adj_array)):
                adj_array.append(padding)

        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg

        if test_acc_avg > best_test_acc:
            best_test_acc = test_acc_avg

        print('best val. acc: {:.4f}, best test acc {:.4f}'.format(best_val_acc, best_test_acc))

        yield {i: # unique_hash
                   {'test_accuracy': test_acc_avg,
                    'validation_accuracy': val_acc_avg,
                    'module_adjacency':adj_array,
                    'module_operations': ops_array.tolist(),
                    'training_time': training_time_avg}}

        i += 1

def transform_operations(ops):
    transform_dict =  {'input':0, 'conv1x1-bn-relu':1, 'conv3x3-bn-relu':2, 'maxpool3x3':3, 'output':4}
    ops_array = np.zeros([7,5], dtype='int8')
    for row, op in enumerate(ops):
        col = transform_dict[op]
        ops_array[row, col] = 1
    return ops_array


def gen_json_file():
    nasbench = api.NASBench(NASBENCH_TFRECORD)
    nas_gen = gen_data_point(nasbench)
    data_dict = OrderedDict()
    for data_point in nas_gen:
        data_dict.update(data_point)
    with open('data/data.json', 'w') as outfile:
        json.dump(data_dict, outfile)




if __name__ == '__main__':
    gen_json_file()
