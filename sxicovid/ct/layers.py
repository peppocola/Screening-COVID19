import torch


class Projector(torch.nn.Module):
    """
    Channel-wise dimensionality projector.
    """
    def __init__(self, in_features, out_features):
        super(Projector, self).__init__()
        self.proj = torch.nn.Conv2d(
            in_channels=in_features, out_channels=out_features,
            kernel_size=(1, 1), padding=(0, 0), bias=False
        )

    def forward(self, x):
        return self.proj(x)


class LinearAttention1d(torch.nn.Module):
    """
    Linear attention with softmax normalization.
    """
    def __init__(self, in_features):
        super(LinearAttention1d, self).__init__()
        self.gate1 = torch.nn.Linear(
            in_features=in_features, out_features=in_features, bias=True
        )
        self.gate2 = torch.nn.Linear(
            in_features=in_features, out_features=in_features, bias=True
        )
        self.score = torch.nn.Linear(
            in_features=in_features, out_features=1, bias=True
        )

    def forward(self, x, g):
        g = g.unsqueeze(1)
        c = self.score(torch.tanh(self.gate1(x) + self.gate2(g)))
        a = torch.softmax(c, dim=1)
        g = torch.sum(a * x, dim=1)
        return a.squeeze(2), g


class LinearAttention2d(torch.nn.Module):
    """
    Linear attention based on parametrized compatibility score function with softmax normalization.
    """
    def __init__(self, in_features):
        super(LinearAttention2d, self).__init__()
        self.score = torch.nn.Conv2d(
            in_channels=in_features, out_channels=1,
            kernel_size=(1, 1), padding=(0, 0), bias=False
        )

    def forward(self, x, g):
        b, f, h, w = x.size()
        c = self.score(x + g)
        a = torch.softmax(c.view(b, 1, -1), dim=2).view(b, 1, h, w)
        g = torch.mul(a.expand_as(x), x)
        g = g.view(b, f, -1).sum(dim=2)
        return a, g
