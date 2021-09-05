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
            lap = torch.zeros((num_nodes, num_nodes)).float().to(y.device)
            lap[row, col] = edge_weight.float()
            return lap
