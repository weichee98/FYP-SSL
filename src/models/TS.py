import torch


class TSEncoder(torch.nn.Module):

    def __init__(self, output_dim, bidirectional=True, dropout=0):
        super().__init__()
        self.encoder = torch.nn.LSTM(
            input_size=1,
            hidden_size=output_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        _, (z, _) = self.encoder(x)
        z = z.transpose(0, 1)
        z = z.reshape((x.shape[0], -1))
        return z


class TSDecoder(torch.nn.Module):

    def __init__(self, input_dim, dropout=0):
        super().__init__()
        self.decoder = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=1,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, z, seq_len):
        z = z.unsqueeze(1)
        z = z.repeat(1, seq_len, 1)
        x, _ = self.decoder(z)
        x = x.squeeze()
        return x


class TSAE(torch.nn.Module):

    def __init__(self, emb=500, bidirectional=True, dropout=0):
        super().__init__()
        self.encoder = TSEncoder(
            output_dim=emb,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.decoder = TSDecoder(
            input_dim=emb * 2 if bidirectional else emb,
            dropout=dropout
        )

    def forward(self, x):
        z = self.encoder(x)
        w = self.decoder(z, x.size(1))
        return z, w


if __name__ == "__main__":
    x = torch.zeros((264, 100))
    lstm = TSAE()
    z, w = lstm(x)
    print(z.size())
    print(w.size())