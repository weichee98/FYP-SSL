import torch
from torch import nn


class LaplacianRegularization(nn.Module):

    def __init__(self, normalization=None, p=2):
        """
        normalization: str or None
        - None: No normalization
            ``L = D - W``
        - "sym": Symmetric normalization
            ``L = I - D^(-1/2) * W * D^(-1/2)``
        """
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
            raise ValueError(
                "invalid normalization {}" \
                    .format(norm)
            )

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

    def __call__(self, edge_index, edge_weights, y):
        """
        edge_index == (2, num_edges)
        edge_weights == (num_edges,)
        y: predicted labels == (num_nodes, num_classes)
        """
        if y.dim() > 2:
            raise ValueError("y has more than 2 dimensions")
        elif y.dim() == 0:
            raise ValueError("y cannot be a scalar with no dimensions")
        
        row, col = edge_index
        y_r, y_c = y[row].float(), y[col].float()

        if self.normalization == "sym":
            degree = self._degree(edge_index, edge_weights, y)
            degree = torch.sqrt(degree)
        else:
            degree = torch.ones(y.size(0)).float()
        d_r, d_c = degree[row], degree[col]
        if y.dim() == 2:
            d_r = d_r.unsqueeze(dim=1)
            d_c = d_c.unsqueeze(dim=1)

        diff = y_r / d_r - y_c / d_c
        l2_norm = torch.norm(diff, dim=1, p=self.p)
        reg = (l2_norm * edge_weights).mean()
        return reg

    @staticmethod
    def _degree(edge_index, edge_weights, y):
        num_nodes = y.size(0)
        adj = torch.zeros((num_nodes, num_nodes)).float().to(y.device)
        row, col = edge_index
        adj[row, col] = edge_weights.float()
        degree = adj.sum(dim=1)
        return degree
