import torch
import torch.nn.functional as F
import numpy as np


def compute_graph_laplacian(A):
    
    D = torch.diag(A.sum(dim=1))
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    L = torch.eye(A.size(0), device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    return L




def sinkhorn_loss(pred, ref, epsilon=0.1, max_iter=100):

    cost_matrix = torch.cdist(pred, ref, p=2).pow(2)  # [N, M]
    Q = -cost_matrix / epsilon
    Q = Q - Q.max()  # for numerical stability
    K = torch.exp(Q)

    u = torch.ones(pred.shape[0], device=pred.device)
    v = torch.ones(ref.shape[0], device=pred.device)

    for _ in range(max_iter):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)

    T = torch.diag(u) @ K @ torch.diag(v)  # Optimal transport plan
    return torch.sum(T * cost_matrix)



def sampled_graph_laplacian_loss(pred: torch.Tensor,
                                 A: torch.Tensor,
                                 num_samples: int = 50_000) -> torch.Tensor:

    if A.is_sparse:
        idx_i, idx_j = A.indices()
        weights      = A.values()
    else:                                              
        idx_i, idx_j = torch.nonzero(A, as_tuple=True)
        weights      = A[idx_i, idx_j]

    E = idx_i.size(0)
    if E > num_samples:                                   
        perm = torch.randperm(E, device=A.device)[:num_samples]
        idx_i, idx_j, weights = idx_i[perm], idx_j[perm], weights[perm]

    diff = pred[idx_i] - pred[idx_j]                        

    loss = (weights.unsqueeze(1) * diff.pow(2)).mean()


    return loss
