import torch
import torchvision

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls

from sxicovid.ct.layers import Projector, LinearAttention


class CTNet(torchvision.models.ResNet):
    def __init__(self, num_classes=2, embeddings=False, pretrained=True):
        super(CTNet, self).__init__(
            block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 6, 3]
        )
        self.num_classes = num_classes
        self.embeddings = embeddings
        self.out_features = 4096

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
        if not self.embeddings:
            self.fc = torch.nn.Linear(self.out_features, self.num_classes)

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

        # Pass through the linear classifier, if specified
        if not self.embeddings:
            x = self.fc(x)

        # Return the attention map, optionally
        if attention:
            return x, torch.sigmoid(c1), torch.sigmoid(c2)
        return x

    def forward(self, x, attention=False):
        return self._forward_impl(x, attention=attention)


class CTSeqNet(torch.nn.Module):
    CT_RESNET50_ATT2_EMBEDDINGS = 'ct-models/ct-resnet50-att2.pt'

    def __init__(
            self,
            input_size,
            hidden_size=512,
            bidirectional=True,
            num_layers=2,
            dropout=0.5,
            num_classes=2,
            load_embeddings=False
    ):
        super(CTSeqNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.load_embeddings = load_embeddings
        self.out_features = hidden_size * 2 if bidirectional else hidden_size

        # Instantiate the image embeddings model
        if self.load_embeddings:
            self.embeddings = CTNet(embeddings=True, pretrained=False)
            state_dict = torch.load(self.CT_RESNET50_ATT2_EMBEDDINGS)
            self.embeddings.load_state_dict(state_dict, strict=False)
            for param in self.embeddings.parameters():
                param.requires_grad = False
        else:
            self.embeddings = CTNet(embeddings=True, pretrained=True)

        # Instantiate the LSTM model
        self.lstm = torch.nn.LSTM(
            self.embeddings.out_features, self.hidden_size, dropout=self.dropout,
            num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True
        )

        # Instantiate the FC model with a dropout layer
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.out_features, self.num_classes)
        )

    def train(self, mode=True):
        self.training = mode
        self.embeddings.train(mode and not self.load_embeddings)
        self.lstm.train(mode)
        self.fc.train(mode)

    def eval(self):
        self.train(False)

    def forward(self, x):
        # [B, L, 224, 224] -> [B * L, 1, 224, 224]
        x = x.view(-1, 1, 224, 224)

        # [B * L, 1, 224, 224] -> [B * L, 4096]
        x = self.embeddings(x)

        # [B * L, 4096] -> [B, L, 4096]
        x = x.view(-1, self.input_size, self.embeddings.out_features)

        # [B, L, 4096] -> [B, L, H]
        x, _ = self.lstm(x)

        # [B, L, H] -> [B, H]
        x = torch.mean(x, dim=1)

        # [B, H] -> [B, C]
        x = self.fc(x)
        return x
