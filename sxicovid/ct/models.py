import torch
import torchvision

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls

from sxicovid.ct.layers import Projector, LinearAttention

RESNET50_COVIDX_CT_FILEPATH = 'ct-models/ct-resnet50.pt'


class CTNet(torchvision.models.ResNet):
    def __init__(self, num_classes=2, pretrained=True):
        super(CTNet, self).__init__(
            block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 6, 3]
        )

        # Check if use pretrained ResNet50 model (on ImageNet)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
            self.load_state_dict(state_dict)

        # Initialize the projectors
        self.projector1 = Projector(512, 2048)
        self.projector2 = Projector(1024, 2048)

        # Initialize the linear attentions
        self.attention1 = LinearAttention(2048)
        self.attention2 = LinearAttention(2048)

        # Re-instantiate the fully connected layer
        del self.fc
        self.fc = torch.nn.Linear(4096, num_classes)

    def _forward_impl(self, x, attention=False):
        # ResNet50 requires 3-channels input
        x = torch.cat([x, x, x], dim=1)

        # Forward through the input convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Forward through the first ResNet50 layer
        x = self.layer1(x)

        # Forward through the innermost two ResNet50 layers to get local feature tensors
        l1 = self.layer2(x)
        l2 = self.layer3(l1)

        # Forward through the last ResNet50 layer
        x = self.layer4(l2)

        # Forward through the average pooling to get global feature vectors
        g = self.avgpool(x)

        # Forward through the attention layers (note the dimensionality projections)
        c1, g1 = self.attention1(self.projector1(l1), g)
        c2, g2 = self.attention2(self.projector2(l2), g)

        # Concatenate the weighted and normalized compatibility scores
        x = torch.cat([g1, g2], dim=1)

        # Pass through the linear classifier
        x = self.fc(x)

        # Return the attention map, optionally
        if attention:
            return x, torch.sigmoid(c1), torch.sigmoid(c2)
        return x

    def forward(self, x, attention=False):
        return self._forward_impl(x, attention=attention)


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
            self.load_state_dict(state_dict)
        elif pretrained == 'covidx-ct':
            state_dict = torch.load(RESNET50_COVIDX_CT_FILEPATH)
            self.load_state_dict(state_dict)
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
    def __init__(
            self,
            input_size,
            hidden_size=512,
            bidirectional=True,
            num_layers=2,
            dropout=0.5,
            n_classes=2,
            pretrained='imagenet'
    ):
        super(CTSeqNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_classes = n_classes
        self.pretrained = pretrained

        self.lstm_out_features = hidden_size * 2 if bidirectional else hidden_size
        self.embeddings = EmbeddingResNet50(self.input_size, pretrained=self.pretrained)

        if self.pretrained == 'covidx-ct':
            for param in self.embeddings.parameters():
                param.requires_grad = False

        self.lstm = torch.nn.LSTM(
            self.embeddings.out_features, self.hidden_size, dropout=self.dropout,
            num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.lstm_out_features, self.n_classes)
        )

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
