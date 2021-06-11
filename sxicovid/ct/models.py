import torch
import torchvision
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls


class EmbeddingResNet50(torchvision.models.ResNet):
    def __init__(self, input_size, pretrained=False, progress=True):
        super(EmbeddingResNet50, self).__init__(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3]
        )
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=progress)
            self.load_state_dict(state_dict)
        self.input_size = input_size
        self.out_features = 2048
        del self.fc
        self.fc = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(-1, self.input_size, self.embeddings_size)


class CTNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size=256, bidirectional=True, dropout=0.5, n_classes=2, pretrained=True):
        super(CTNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.n_classes = n_classes
        self.pretrained = pretrained

        self.lstm_out_features = hidden_size * 2 if bidirectional else hidden_size
        self.fc_hidden_size = self.lstm_out_features // 4
        self.embeddings = EmbeddingResNet50(self.input_size, pretrained=self.pretrained)

        self.lstm = torch.nn.LSTM(
            self.embeddings.out_features, self.hidden_size,
            bidirectional=self.bidirectional, batch_first=True
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.lstm_out_features, self.fc_hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.input_size * self.fc_hidden_size, self.n_classes)
        )

    def forward(self, x):
        # [B, L, 224, 224] -> [B * L, 224, 224]
        x = x.reshape([-1, 224, 224])

        # [B * L, 224, 224] -> [B * L, 3, 224, 224]
        x = torch.stack([x, x, x], dim=1)

        # [B * L, 3, 224, 224] -> [B, L, S]
        x = self.embeddings(x)

        # [B, L, S] -> [B, L, H]
        x, _ = self.lstm(x)

        # [B, L, H] -> [B, C]
        x = self.fc(x)
        return x
