import json
import os
import numpy as np
#from prettytable import PrettyTable


#t = PrettyTable(['Method', 'CIFAR-10 val', 'CIFAR-10 test', 'CIFAR-100 val', 'CIFAR-100 test', 'ImageNet-16-120 val', 'ImageNet-16-120 test'])
#t = PrettyTable(['Method', 'CIFAR-10 val', 'CIFAR-10 test'])

def get_summary(dataset, file_name, data_dir, val_test, N_runs):
    val_acc = []
    test_acc = []
    for k in range(1, N_runs+1):
        file_name_ = file_name.format(dataset, k)
        file_path = os.path.join(data_dir, file_name_)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                acc_dict = json.load(f)
                val_acc.append(acc_dict[val_test[0]]) # using average instead of individual
                test_acc.append(acc_dict[val_test[1]])
    val_acc = np.array(val_acc)
    test_acc = np.array(test_acc)

    return val_acc.mean(), val_acc.std(), test_acc.mean(), test_acc.std()


# RL (ours)
row = ['arch2vec-RL']
data_dir = 'saved_logs/rl/dim16/'
datasets = {'cifar10_valid_converged':500, 'cifar100':500, 'ImageNet16_120':500}
file_name = 'nasbench201_{}_run_{}_full.json'
val_test = ['val_acc_avg', 'test_acc_avg']
for i, (dataset, N_runs) in enumerate(datasets.items()):
    val_mean, val_std, test_mean, test_std = get_summary(dataset, file_name, data_dir, val_test, N_runs)
    row.append('{:.2f}+-{:.2f}'.format(val_mean, val_std))
    row.append('{:.2f}+-{:.2f}'.format(test_mean, test_std))
print(row)



## BO (ours)
row = ['arch2vec-BO']
data_dir = 'saved_logs/bo/dim16/'
datasets = {'cifar10_valid_converged':500, 'cifar100':500, 'ImageNet16_120':500}
file_name = 'nasbench201_{}_run_{}_full.json'
val_test = ['val_acc_avg', 'test_acc_avg']
for i, (dataset, N_runs) in enumerate(datasets.items()):
    val_mean, val_std, test_mean, test_std = get_summary(dataset, file_name, data_dir, val_test, N_runs)
    row.append('{:.2f}+-{:.2f}'.format(val_mean, val_std))
    row.append('{:.2f}+-{:.2f}'.format(test_mean, test_std))


print(row)




