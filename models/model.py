import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution
from utils.utils import preprocessing, normalize_adj
import time
from gin.models.mlp import MLP


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, dropout, **kwargs):
        super(GIN, self).__init__()
        self.num_layers = num_hops
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, input_dim, dropout, **kwargs)

    def _encoder(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj, x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        x = self.fc(x)
        return x


    def forward(self, ops, adj):
        x = self._encoder(ops, adj)
        ops_recon, adj_recon = self.decoder(x)
        return ops_recon, adj_recon, x

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers,
                 dropout, **kwargs):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.input_dim, dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _encoder(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar

    def forward(self, ops, adj):
        mu, logvar = self._encoder(ops, adj)
        z = self.reparameterize(mu, logvar)
        ops_recon, adj_recon = self.decoder(z)
        return ops_recon, adj_recon, mu, logvar

class GAE(nn.Module):
    def __init__(self, dims, normalize, reg_emb, reg_dec_l2, reg_dec_gp, dropout, **kwargs):
        super(GAE, self).__init__()
        self.encoder = Encoder(dims, normalize, reg_emb, dropout)
        self.decoder = Decoder(dims[-1], dims[0], dropout, **kwargs)
        self.reg_dec_l2 = reg_dec_l2
        self.reg_dec_gp = reg_dec_gp

    def forward(self, ops, adj):
        x, emb_loss = self.encoder(ops, adj)
        ops_recon, adj_recon = self.decoder(x)
        if self.reg_dec_l2:
            dec_loss_l2 = 0
            for p in self.decoder.parameters():
                dec_loss_l2 += torch.norm(p, 2)
            return ops_recon, adj_recon, emb_loss, dec_loss_l2, None
        if self.reg_dec_gp:
            return ops_recon, adj_recon, emb_loss, torch.FloatTensor([0.]).cuda(), x
        return ops_recon, adj_recon, emb_loss, torch.FloatTensor([0.]).cuda(), None

class GVAE(nn.Module):
    def __init__(self, dims, normalize, dropout, **kwargs):
        super(GVAE, self).__init__()
        self.encoder = VAEncoder(dims, normalize, dropout)
        self.decoder = Decoder(dims[-1], dims[0], dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, ops, adj):
        mu, logvar = self.encoder(ops, adj)
        z = self.reparameterize(mu, logvar)
        ops_recon, adj_recon = self.decoder(z)
        return ops_recon, adj_recon, mu, logvar

class Encoder(nn.Module):
    def __init__(self, dims, normalize, reg_emb, dropout):
        super(Encoder, self).__init__()
        self.gcs = nn.ModuleList(self.get_gcs(dims, dropout))
        self.normalize = normalize
        self.reg_emb = reg_emb

    def get_gcs(self,dims,dropout):
        gcs = []
        for k in range(len(dims)-1):
            gcs.append(GraphConvolution(dims[k],dims[k+1], dropout))
        return gcs

    def forward(self, ops, adj):
        if self.normalize:
            adj = normalize_adj(adj)
        x = ops
        for gc in self.gcs:
            x = gc(x, adj)
        if self.reg_emb:
            emb = x.mean(dim=1).squeeze()
            emb_loss = torch.mean(torch.norm(emb, p=2, dim=1))
            return x, emb_loss
        return x, torch.FloatTensor([0.]).cuda()

class VAEncoder(nn.Module):
    def __init__(self, dims, normalize, dropout):
        super(VAEncoder, self).__init__()
        self.gcs = nn.ModuleList(self.get_gcs(dims, dropout))
        self.gc_mu = GraphConvolution(dims[-2], dims[-1], dropout)
        self.gc_logvar = GraphConvolution(dims[-2], dims[-1], dropout)
        self.normalize = normalize

    def get_gcs(self,dims,dropout):
        gcs = []
        for k in range(len(dims)-1):
            gcs.append(GraphConvolution(dims[k],dims[k+1], dropout))
        return gcs

    def forward(self, ops, adj):
        if self.normalize:
            adj = normalize_adj(adj)
        x = ops
        for gc in self.gcs[:-1]:
            x = gc(x, adj)
        mu = self.gc_mu(x, adj)
        logvar = self.gc_logvar(x, adj)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_dim, dropout, activation_adj=torch.sigmoid, activation_ops=torch.sigmoid, adj_hidden_dim=None, ops_hidden_dim=None):
        super(Decoder, self).__init__()
        if adj_hidden_dim == None:
            self.adj_hidden_dim = embedding_dim
        if ops_hidden_dim == None:
            self.ops_hidden_dim = embedding_dim
        self.activation_adj = activation_adj
        self.activation_ops = activation_ops
        self.weight = torch.nn.Linear(embedding_dim, input_dim)
        self.dropout = dropout

    def forward(self, embedding):
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        ops = self.weight(embedding)
        adj = torch.matmul(embedding, embedding.permute(0, 2, 1))
        return self.activation_adj(ops), self.activation_adj(adj)

class Reconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        return loss


class VAEReconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + KLD


class WeightedLoss(nn.MSELoss):
    def __init__(self, weight=None):
        super(WeightedLoss, self).__init__()
        self.weight = weight


    def forward(self, inputs, targets):
        res = (torch.exp(inputs)-1.0) * F.mse_loss(inputs, targets, size_average=False)
        return torch.mean(res, dim=0) / (self.weight - 1)



class LinearModel(nn.Module):
    def __init__(self, input_dim, hid_dim, activation=F.relu):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)
        self.activation = activation

    def forward(self, x):
        h = self.activation(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        return y

class DecoderNN(object):
    def __init__(self, model, ops, adj, cfg):
        print('Initializing NN decoder')
        t_s = time.time()
        self.model = model
        self.ops = ops
        self.adj = adj
        self.cfg = cfg
        with torch.no_grad():
            adj_prep, ops_prep, _ = preprocessing(self.adj, self.ops, **self.cfg['prep'])
            self.embedding = self.model.encoder(ops_prep, adj_prep)
        assert len(self.embedding.shape) == 3
        print('Using {} seconds to initialize NN decoder'.format(time.time()-t_s))

    def find_NN(self, ops, adj, ind, k = 10):
        assert len(ops.shape)==3
        ind_t1_list = []
        ind_tk_list = []
        with torch.no_grad():
            adj_prep, ops_prep, _ = preprocessing(adj, ops, **self.cfg['prep'])
            embeddings = self.model.encoder(ops_prep, adj_prep)
            for e in embeddings:
                dist = torch.sum( (self.embedding - e) ** 2, dim=[1,2])
                _, ind_t1 = torch.topk(dist, 1, largest=False)
                _, ind_tk = torch.topk(dist, k, largest=False)
                ind_t1_list.append(ind_t1.item())
                ind_tk_list.append(ind_tk.tolist())
        op_recon, adj_recon = self.ops[ind_t1_list], self.adj[ind_t1_list]
        op_recon_tk, adj_recon_tk = self.ops[ind_t1_list], self.adj[ind_t1_list]
        return op_recon, adj_recon, op_recon_tk, adj_recon_tk, ind_t1_list, ind_tk_list
