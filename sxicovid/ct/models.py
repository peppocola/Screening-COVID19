import torch
import torchvision

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls

RESNET50_COVIDX_CT_FILEPATH = 'ct-models/ct-resnet50.pt'


class CTNet(torch.nn.Module):
    def __init__(self, base, n_classes=2, pretrained=True):
        super(CTNet, self).__init__()
        self.network = CTNet.__build_network(base, num_classes=n_classes, pretrained=pretrained)

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
    def __init__(self, input_size, pretrained='imagenet', progress=True):
        num_classes = 3 if pretrained == 'covidx-ct' else 1000
        super(EmbeddingResNet50, self).__init__(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes
        )
        self.input_size = input_size
        self.out_features = 2048

        # Load the pretrained resnet50
        if pretrained == 'none':
            pass
        elif pretrained == 'imagenet':
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=progress)
            self.load_state_dict(state_dict, strict=True)
        elif pretrained == 'covidx-ct':
            state_dict = torch.load(RESNET50_COVIDX_CT_FILEPATH)
            self.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError('Unknown pretrained value {}'.format(pretrained))

        # Delete the fully connected layer
        del self.fc

    def forward(self, x):
        # [B, L, 224, 224] -> [B * L, 224, 224]
        x = x.view(-1, 224, 224)

        # [B * L, 3, 224, 224]
        x = torch.stack([x, x, x], dim=1)

        # Apply ResNet features extractor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # [B * L, 2048] -> [B, L, 2048]
        x = x.view(-1, self.input_size, self.out_features)
        return x


class CTSeqNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size=512, bidirectional=True, num_layers=2, n_classes=2, pretrained='imagenet'):
        super(CTSeqNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.pretrained = pretrained

        self.lstm_out_features = hidden_size * 2 if bidirectional else hidden_size
        self.embeddings = EmbeddingResNet50(self.input_size, pretrained=self.pretrained)

        if self.pretrained == 'covidx-ct':
            for param in self.embeddings.parameters():
                param.requires_grad = False

        self.lstm = torch.nn.LSTM(
            self.embeddings.out_features, self.hidden_size,
            num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.fc = torch.nn.Linear(self.lstm_out_features, self.n_classes)

    def train(self, mode=True):
        self.training = mode
        self.embeddings.train(mode and self.pretrained != 'covidx-ct')
        self.lstm.train(mode)
        self.fc.train(mode)

    def eval(self):
        self.train(False)

    def forward(self, x):
        # [B, L, 224, 224] -> [B, L, S]
        x = self.embeddings(x)

        # [B, L, S] -> [B, L, H]
        x, _ = self.lstm(x)

        # [B, L, H] -> [B, H]
        x = torch.mean(x, dim=1)

        # [B, H] -> [B, C]
        x = self.fc(x)
        return x
