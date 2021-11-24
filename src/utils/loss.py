import torch
from torch import nn
from torch_geometric.utils import get_laplacian


def reduce(x, reduction="none"):
    if reduction == "none":
        return x
    elif reduction == "mean":
        return x.mean()
    elif reduction == "sum":
        return x.sum()
    else:
        raise TypeError("invalid reduction: {}".format(reduction))


class LaplacianRegularization(nn.Module):

    def __init__(self, normalization=None, p=2):
        """
        normalization: str or None
        - None: No normalization
            ``L = D - W``
        - "sym": Symmetric normalization
            ``L = I - D^(-1/2) * W * D^(-1/2)``
        - "rw": Random-walk normalization
            ``L = I - D^(-1) * A``
        """
        super().__init__()
        self.normalization = normalization
        self.p = p

    @property
    def normalization(self):
        return self._norm

    @normalization.setter
    def normalization(self, norm):
        if norm in [None, "sym"]:
            self._norm = norm
        else:
            raise ValueError("invalid normalization {}".format(norm))

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        if isinstance(p, int):
            if p > 0:
                self._p = p
            else:
                raise ValueError("p must be greater than 0")
        else:
            raise TypeError("p must be an integer")

    def forward(self, edge_index, edge_weights, y):
        """
        edge_index == (2, num_edges)
        edge_weights == (num_edges,)
        y: predicted labels == (num_nodes, num_classes)
        """        
        lap = self._laplacian(edge_index, edge_weights, y)
        reg = torch.matmul(torch.matmul(y.T, lap), y)
        reg = torch.diagonal(reg).mean()
        return reg

    def _laplacian(self, edge_index, edge_weight, y):
        with torch.no_grad():
            edge_index, edge_weight = get_laplacian(
                edge_index, edge_weight, self.normalization
            )
            num_nodes = y.size(0)
            row, col = edge_index
            lap = torch.zeros((num_nodes, num_nodes), dtype=edge_weight.dtype).to(y.device)
            lap[row, col] = edge_weight
            return lap


class GaussianKLDivLoss(nn.Module):
    """
    closed form solution for kl divergence if 2 distributions are Gaussians
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu1, var1, mu2, var2):
        """
        KL(p1 | p2)
        p1 ~ N(mu1, var1)
        p2 ~ N(mu2, var2)
        """
        kl = 0.5 * (
            var2.log() - var1.log() + \
            (var1 + (mu1 - mu2) ** 2) / var2 - 1
        )
        kl = kl.sum(dim=1)
        return reduce(kl, self.reduction)