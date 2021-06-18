import torch


class LinearAttention1d(torch.nn.Module):
    """
    Linear attention with softmax normalization.
    """
    def __init__(self, in_features):
        super(LinearAttention1d, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.gate1 = torch.nn.Linear(
            in_features=self.in_features, out_features=self.in_features, bias=False
        )
        self.gate2 = torch.nn.Linear(
            in_features=self.in_features, out_features=self.in_features, bias=True
        )
        self.score = torch.nn.Linear(
            in_features=self.in_features, out_features=1, bias=True
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
    def __init__(self, in_features, out_features):
        super(LinearAttention2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj = torch.nn.Conv2d(
            in_channels=self.in_features, out_channels=self.out_features,
            kernel_size=(1, 1), padding=(0, 0), bias=False
        )
        self.score = torch.nn.Conv2d(
            in_channels=self.out_features, out_channels=1,
            kernel_size=(1, 1), padding=(0, 0), bias=False
        )

    def forward(self, x, g):
        b, _, h, w = x.size()
        c = self.score(x + self.proj(g))
        a = torch.softmax(c.view(b, 1, -1), dim=2).view(b, 1, h, w)
        g = torch.mul(a.expand_as(x), x)
        g = g.view(b, self.out_features, -1).sum(dim=2)
        return a, g
