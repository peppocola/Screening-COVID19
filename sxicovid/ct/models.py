import torch
import torchvision

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls


class CTNet(torch.nn.Module):
    def __init__(self, base, num_classes=2, pretrained=True):
        super(CTNet, self).__init__()
        self.network = CTNet.__build_network(base, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)
        return self.network(x)

    @staticmethod
    def __build_network(base, num_classes=1000, pretrained=True):
        if base == 'resnet50':
            network = torchvision.models.resnet50(pretrained=pretrained)
            network.fc = torch.nn.Linear(2048, num_classes)
        elif base == 'densenet121':
            network = torchvision.models.densenet121(pretrained=pretrained)
            network.classifier = torch.nn.Linear(1024, num_classes)
        elif base == 'inceptionv3':
            network = torch.nn.Sequential(
                torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True),
                torchvision.models.inception_v3(
                    pretrained=False, num_classes=num_classes, aux_logits=True, init_weights=True
                )
            )
        else:
            raise NotImplementedError('Unknown base model {}'.format(base))
        return network


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
        x = x.view(-1, 224, 224)
        x = torch.stack([x, x, x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(-1, self.input_size, self.out_features)
        return x


class CTSeqNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, bidirectional=True, num_layers=2, n_classes=2, pretrained=False):
        super(CTSeqNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.pretrained = pretrained

        self.lstm_out_features = num_layers * (hidden_size * 2 if bidirectional else hidden_size)
        self.embeddings = EmbeddingResNet50(self.input_size, pretrained=self.pretrained)

        self.lstm = torch.nn.LSTM(
            self.embeddings.out_features, self.hidden_size,
            num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.fc = torch.nn.Linear(self.lstm_out_features, self.n_classes)

    def forward(self, x):
        # [B, L, 224, 224] -> [B, L, S]
        x = self.embeddings(x)

        # [B, L, S] -> [B, H]
        _, (x, _) = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = torch.flatten(x, 1)

        # [B, H] -> [B, C]
        x = self.fc(x)
        return x
