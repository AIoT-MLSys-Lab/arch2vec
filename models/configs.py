import torch
import torch.nn as nn
import torch.nn.functional as F

configs = [{'GAE': # 0
                {'activation_ops':torch.sigmoid},
            'loss':
                {'loss_ops':F.mse_loss, 'loss_adj':F.mse_loss},
            'prep':
                {'method':3, 'lbd':0.5}
            },
           {'GAE': # 1
                {'activation_ops':torch.softmax},
            'loss':
                {'loss_ops':nn.BCELoss(), 'loss_adj':nn.BCELoss()},
            'prep':
                {'method':3, 'lbd':0.5}
            },
           {'GAE': # 2
                {'activation_ops': torch.softmax},
            'loss':
                {'loss_ops': F.mse_loss, 'loss_adj': nn.BCELoss()},
            'prep':
                {'method':3, 'lbd':0.5}
            },
           {'GAE':# 3
                {'activation_ops':torch.sigmoid},
            'loss':
                {'loss_ops':F.mse_loss, 'loss_adj':F.mse_loss},
            'prep':
                {'method':4, 'lbd':1.0}
            },
           {'GAE': # 4
                {'activation_adj': torch.sigmoid, 'activation_ops': torch.softmax, 'adj_hidden_dim': 128, 'ops_hidden_dim': 128},
            'loss':
                {'loss_ops': nn.BCELoss(), 'loss_adj': nn.BCELoss()},
            'prep':
                {'method': 4, 'lbd': 1.0}
            },
           ]
