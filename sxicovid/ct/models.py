import torch


class CTNet(torch.nn.Module):
    def __init__(self, ct_size, hidden_size=64, avgpool_size=1, n_classes=2):
        super(CTNet, self).__init__()
        self.ct_size = ct_size
        self.hidden_size = hidden_size
        self.avgpool_size = avgpool_size
        self.n_classes = n_classes

        assert 512 % self.ct_size == 0, '512 must be divisible by ct_size'
        self.embedding_size = (512 // self.ct_size) * self.avgpool_size * self.avgpool_size

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.ct_size, 64, 5, stride=2, padding=2, groups=self.ct_size, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 3, padding=1, groups=self.ct_size, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(128, 256, 3, padding=1, groups=self.ct_size, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(256, 512, 3, padding=1, groups=self.ct_size, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d((2, 2))
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((self.avgpool_size, self.avgpool_size))
        self.convlstm = torch.nn.Sequential(
            torch.nn.Conv1d(self.embedding_size, self.embedding_size, 3, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(self.embedding_size, self.embedding_size, 3, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.LSTM(self.ct_size, self.hidden_size, batch_first=True),
        )
        self.fc = torch.nn.Linear(self.embedding_size * self.hidden_size, self.n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape([-1, self.ct_size, self.embedding_size]).transpose(2, 1)
        x, _ = self.convlstm(x)
        x = x.reshape([-1, self.embedding_size * self.hidden_size])
        x = self.fc(x)
        return x
