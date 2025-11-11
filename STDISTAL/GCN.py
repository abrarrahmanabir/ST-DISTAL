from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import time
import multiprocessing
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy
from STDISTAL.losses import sinkhorn_loss, sampled_graph_laplacian_loss
import time, copy
import torch
import math
from torch.nn import Parameter, Module



def compute_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:

    assert adj.size(0) == adj.size(1), "Adjacency matrix must be square"
    N = adj.size(0)

    if not adj.is_sparse:
        adj = adj.to_sparse()

    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1e-6)  # [N]
    deg_inv_sqrt = deg.pow(-0.5)                                   # [N]
    
    idx = torch.arange(N, device=adj.device)
    D_inv_sqrt = torch.sparse_coo_tensor(
        torch.stack([idx, idx]),
        deg_inv_sqrt,
        size=(N, N)
    )

    norm_adj = torch.sparse.mm(D_inv_sqrt, torch.sparse.mm(adj, D_inv_sqrt))
    eye_idx = torch.arange(N, device=adj.device)
    I = torch.sparse_coo_tensor(
        indices=torch.stack([eye_idx, eye_idx]),
        values=torch.ones(N, device=adj.device),
        size=(N, N)
    )

    # Equation 3
    L = I - norm_adj
    return L


class conGraphConvolutionlayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, cheb_order=2, gamma=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cheb_order = cheb_order
        self.gamma = gamma


        self.gcn_weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        

        self.cheb_weight = nn.Parameter(torch.FloatTensor(cheb_order + 1, in_features, out_features))

        self.alpha_cheb = nn.Parameter(torch.tensor(5.0), requires_grad=True) 
        self.gamma_param = nn.Parameter(torch.tensor(gamma), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gcn_weight)
        nn.init.xavier_uniform_(self.cheb_weight)
        if self.bias is not None:
            fan_in = self.gcn_weight.size(0)
            bound = 1 / math.sqrt(fan_in)
            self.bias.data.uniform_(-bound, bound)

    def forward(self, x, adj):
        N = adj.size(0)
        device = x.device

        support_gcn = torch.matmul(x, self.gcn_weight)

        # Equation 2
        output_gcn = torch.spmm(adj, support_gcn)


        L = compute_normalized_laplacian(adj)
        L_hat = L * 1.0 - torch.eye(N, device=device).to_sparse()

        # Equation 5, 6, 7
        T_k = [x]
        if self.cheb_order > 0:
            T_k.append(torch.sparse.mm(L_hat, x))
            for k in range(2, self.cheb_order + 1):
                T_k.append(2 * torch.sparse.mm(L_hat, T_k[-1]) - T_k[-2])
        basis = torch.stack(T_k, dim=0)

        norm_cheb_weight = self.cheb_weight + self.gamma_param * torch.norm(self.cheb_weight, dim=(1, 2), keepdim=True)

        # Equation 8
        output_cheb = torch.einsum('kni,kio->no', basis, norm_cheb_weight) * 1e-3


        a = torch.sigmoid(self.alpha_cheb)

        # Equation 9
        output = a * output_gcn + (1 - a) * output_cheb 

        if self.bias is not None:
            output += self.bias

        return output



class conGCN(nn.Module):
    def __init__(self, nfeat, 
                 nhid, 
                 common_hid_layers_num, 
                 fcnn_hid_layers_num, 
                 dropout, 
                 nout1, 
                ):
        super(conGCN, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.common_hid_layers_num = common_hid_layers_num
        self.fcnn_hid_layers_num = fcnn_hid_layers_num
        self.nout1 = nout1
        self.dropout = dropout
        self.training = True
        

        self.gc_in_exp = conGraphConvolutionlayer(nfeat, nhid)
        self.bn_node_in_exp = nn.BatchNorm1d(nhid)
        self.gc_in_sp = conGraphConvolutionlayer(nfeat, nhid)
        self.bn_node_in_sp = nn.BatchNorm1d(nhid)
        

        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec('self.cgc{}_exp = conGraphConvolutionlayer(nhid, nhid)'.format(i+1))
                exec('self.bn_node_chid{}_exp = nn.BatchNorm1d(nhid)'.format(i+1))
                exec('self.cgc{}_sp = conGraphConvolutionlayer(nhid, nhid)'.format(i+1))
                exec('self.bn_node_chid{}_sp = nn.BatchNorm1d(nhid)'.format(i+1))


        self.gc_out11 = nn.Linear(2*nhid, nhid, bias=True)
        self.bn_out1 = nn.BatchNorm1d(nhid)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec('self.gc_out11{} = nn.Linear(nhid, nhid, bias=True)'.format(i+1))
                exec('self.bn_out11{} = nn.BatchNorm1d(nhid)'.format(i+1))
        self.gc_out12 = nn.Linear(nhid, nout1, bias=True)
        

    def forward(self, x, adjs):    
        
        self.x = x
        

        self.x_exp = self.gc_in_exp(self.x, adjs[0])
        self.x_exp = self.bn_node_in_exp(self.x_exp)
        self.x_exp = F.elu(self.x_exp)
        self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        self.x_sp = self.gc_in_sp(self.x, adjs[1])
        self.x_sp = self.bn_node_in_sp(self.x_sp)
        self.x_sp = F.elu(self.x_sp)
        self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)
        

        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("self.x_exp = self.cgc{}_exp(self.x_exp, adjs[0])".format(i+1))
                exec("self.x_exp = self.bn_node_chid{}_exp(self.x_exp)".format(i+1))
                self.x_exp = F.elu(self.x_exp)
                self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
                exec("self.x_sp = self.cgc{}_sp(self.x_sp, adjs[1])".format(i+1))
                exec("self.x_sp = self.bn_node_chid{}_sp(self.x_sp)".format(i+1))
                self.x_sp = F.elu(self.x_sp)
                self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        # Equation 10
        self.x1 = torch.cat([self.x_exp, self.x_sp], dim=1)
        self.x1 = self.gc_out11(self.x1)
        self.x1 = self.bn_out1(self.x1)
        self.x1 = F.elu(self.x1)
        self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec("self.x1 = self.gc_out11{}(self.x1)".format(i+1))
                exec("self.x1 = self.bn_out11{}(self.x1)".format(i+1))
                self.x1 = F.elu(self.x1)
                self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        self.x1 = self.gc_out12(self.x1)

        gc_list = {}
        gc_list['gc_in_exp'] = self.gc_in_exp
        gc_list['gc_in_sp'] = self.gc_in_sp
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("gc_list['cgc{}_exp'] = self.cgc{}_exp".format(i+1, i+1))
                exec("gc_list['cgc{}_sp'] = self.cgc{}_sp".format(i+1, i+1))
        gc_list['gc_out11'] = self.gc_out11
        if self.fcnn_hid_layers_num > 0:
            exec("gc_list['gc_out11{}'] =  self.gc_out11{}".format(i+1, i+1))
        gc_list['gc_out12'] = self.gc_out12
        
        return F.log_softmax(self.x1, dim=1), gc_list


def conGCN_train(model, 
                 train_valid_len,
                 test_len, 
                 feature, 
                 adjs, 
                 label, 
                 epoch_n, 
                 loss_fn, 
                 optimizer, 
                 train_valid_ratio=0.9,
                 scheduler=None,
                 early_stopping_patience=5,
                 clip_grad_max_norm=1,
                 load_test_groundtruth=False,
                 print_epoch_step=1,
                 cpu_num=-1,
                 ot_batch_size = 1000,
                 GCN_device='CPU',

                 # loss settings
                 lambda_ot=0.000001,
                 lambda_lap=0.000001,
                 sinkhorn_eps= 0.05, 
                 sinkhorn_iter=50
                ):


    if GCN_device == 'CPU':
        device = torch.device("cpu")
        print('Use CPU as device.')
    else:
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(f'Use {device} as device.')

    if cpu_num == -1:
        torch.set_num_threads(multiprocessing.cpu_count())
    else:
        torch.set_num_threads(cpu_num)

    model = model.to(device)
    adjs = [adj.to(device) for adj in adjs]
    feature = feature.to(device)
    label = label.to(device)

    train_idx = range(int(train_valid_len * train_valid_ratio))
    valid_idx = range(len(train_idx), train_valid_len)

    best_val = float('inf')
    clip = 0
    loss = []
    para_list = []

    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass

        optimizer.zero_grad()
        output1, paras = model(feature.float(), adjs)
        prob = output1.exp()

        # --- original KL losses

        # Equation 11
        loss_train1 = loss_fn(output1[list(np.array(train_idx)+test_len)],
                              label[list(np.array(train_idx)+test_len)].float())
        loss_val1 = loss_fn(output1[list(np.array(valid_idx)+test_len)],
                            label[list(np.array(valid_idx)+test_len)].float())

        if load_test_groundtruth:
            loss_test1 = loss_fn(output1[:test_len], label[:test_len].float())
            loss.append([loss_train1.item(), loss_val1.item(), loss_test1.item()])
        else:
            loss.append([loss_train1.item(), loss_val1.item(), None])

        # OT + Laplacian loss

        n_nodes = prob.size(0)
        if 0 < ot_batch_size < n_nodes:
            sel = torch.randperm(n_nodes, device=prob.device)[:ot_batch_size]
            p_sel = prob[sel]
            l_sel = label[sel].float()
        else:
            p_sel = prob
            l_sel = label.float()

        # Equation 13
        loss_ot_raw = sinkhorn_loss(p_sel, l_sel, epsilon=sinkhorn_eps, max_iter=sinkhorn_iter)
        loss_ot = lambda_ot * loss_ot_raw

        # Equation 15
        loss_lap_raw = sampled_graph_laplacian_loss(prob, adjs[1])
        loss_lap = lambda_lap * loss_lap_raw

        # Equation 16
        total_loss = loss_train1 + loss_ot  + loss_lap


        if epoch % print_epoch_step == 0:
            print("******************************************")
            print(f"Epoch {epoch+1}/{epoch_n}  loss_train: {loss_train1.item()}  "
                  f"loss_val: {loss_val1.item()}  OT: {loss_ot.item()}  Lap: {loss_lap.item()}",
                  end='\t')
            if load_test_groundtruth:
                print(f"Test loss= {loss_test1.item():.4f}", end='\t')
            print(f'time: {time.time():.2f}s')

        para_list.append(paras.copy())
        for i in paras.keys():
            para_list[-1][i] = copy.deepcopy(para_list[-1][i])

        if early_stopping_patience > 0:
            if torch.round(loss_val1, decimals=4) < best_val:
                best_val = torch.round(loss_val1, decimals=4)
                best_paras = paras.copy()
                best_loss = loss.copy()
                clip = 1
                for i in paras.keys():
                    best_paras[i] = copy.deepcopy(best_paras[i])
            else:
                clip += 1
                if clip == early_stopping_patience:
                    break
        else:
            best_loss = loss.copy()
            best_paras = None

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
        optimizer.step()
        if scheduler is not None:
            try:
                scheduler.step()
            except:
                scheduler.step(metrics=loss_val1)

    print("***********************Final Loss***********************")
    print(f"Epoch {epoch+1}/{epoch_n}  loss_train: {loss_train1.item():.4f}  "
          f"loss_val: {loss_val1.item():.4f}", end='\t')
    if load_test_groundtruth:
        print(f"Test loss= {loss_test1.item():.4f}", end='\t')
    print()

    torch.cuda.empty_cache()
    return output1.cpu(), loss, model.cpu()










