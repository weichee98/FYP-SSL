from torch import nn


class PAE(nn.Module):

    def __init__(self, input_size, hidden=128, dropout=0.2):
        super(PAE, self).__init__()
        self.parser = nn.Sequential(
            nn.Linear(input_size, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden, bias=True),
        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, x, edge_index):
        """
        x: (num_samples, num_nonimg_feature)
        edge_index: (2, num_edges)
        """
        x1 = x[:, edge_index[0]]
        x2 = x[:, edge_index[1]]
        h1 = self.parser(x1) 
        h2 = self.parser(x2) 
        edge_weight = (self.cos(h1, h2) + 1) * 0.5
        return edge_weight
