import torch
import torchvision

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls

from sxicovid.ct.layers import Projector
from sxicovid.ct.layers import LinearAttention1d, LinearAttention2d


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
        self.attention1 = LinearAttention2d(2048)
        self.attention2 = LinearAttention2d(2048)

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
        a1, g1 = self.attention1(self.projector1(l1), g)
        a2, g2 = self.attention2(self.projector2(l2), g)

        # Concatenate the weighted and normalized compatibility scores
        x = torch.cat([g1, g2], dim=1)

        # Pass through the linear classifier, if specified
        if not self.embeddings:
            x = self.fc(x)

        # Return the attention map, optionally
        if attention:
            return x, a1, a2
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
            num_classes=2,
            load_embeddings=False
    ):
        super(CTSeqNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
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
            self.embeddings.out_features, self.hidden_size,
            num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True
        )

        # Instantiate the FC model
        self.fc = torch.nn.Linear(self.out_features, self.num_classes)

        # Instantiate the attention module
        self.attention = LinearAttention1d(self.out_features)

    def train(self, mode=True):
        self.training = mode
        self.embeddings.train(mode and not self.load_embeddings)
        self.lstm.train(mode)
        self.fc.train(mode)

    def forward(self, x, attention=False):
        # Squeeze along the batch size
        # [B, L, 224, 224] -> [B * L, 1, 224, 224]
        x = x.view(-1, 1, 224, 224)

        # Obtain the embeddings and the related attention maps, if specified
        # [B * L, 1, 224, 224] -> [B * L, 4096]
        if attention:
            x, e1, e2 = self.embeddings(x, attention=True)
        else:
            x = self.embeddings(x, attention=False)

        # Un-squeeze along the batch size
        # [B * L, 4096] -> [B, L, 4096]
        x = x.view(-1, self.input_size, self.embeddings.out_features)

        # Pass through the LSTM module
        # [B, L, 4096] -> [B, L, H]
        x, _ = self.lstm(x)
        g = x[:, -1]

        # Pass through the attention module
        a, g = self.attention(x.permute(0, 2, 1), g.unsqueeze(2))

        # Pass through the linear classifier
        # [B, H] -> [B, C]
        x = self.fc(g)
        if attention:
            # Un-squeeze the attention maps along the batch size
            e1h, e1w = e1.shape[2], e1.shape[3]
            e2h, e2w = e2.shape[2], e2.shape[3]
            e1 = e1.view(-1, self.input_size, e1h, e1w)
            e2 = e2.view(-1, self.input_size, e2h, e2w)
            return x, a, e1, e2
        return x
