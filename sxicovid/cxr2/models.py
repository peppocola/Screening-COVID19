import torch
import torchvision


class CXR2Net(torch.nn.Module):
    def __init__(self, base='alexnet'):
        super(CXR2Net, self).__init__()
        self.network = CXR2Net.__build_network(base)

    def forward(self, x):
        x = x.repeat([1, 3, 1, 1])
        return self.network(x)

    @staticmethod
    def __build_network(base):
        if base == 'alexnet':
            return torchvision.models.alexnet(pretrained=False, num_classes=2)
        elif base == 'vgg11':
            network = torchvision.models.vgg11(pretrained=False, num_classes=2)
        elif base == 'vgg13':
            network = torchvision.models.vgg13(pretrained=False, num_classes=2)
        elif base == 'vgg16':
            network = torchvision.models.vgg16(pretrained=False, num_classes=2)
        elif base == 'resnet18':
            network = torchvision.models.resnet18(pretrained=False, num_classes=2)
        elif base == 'resnet34':
            network = torchvision.models.resnet34(pretrained=False, num_classes=2)
        elif base == 'resnet50':
            network = torchvision.models.resnet50(pretrained=False, num_classes=2)
        elif base == 'densenet121':
            network = torchvision.models.densenet121(pretrained=False, num_classes=2)
        elif base == 'inceptionv3':
            network = torch.nn.Sequential(
                torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True),
                torchvision.models.inception_v3(pretrained=False, num_classes=2, aux_logits=True, init_weights=True)
            )
        else:
            raise NotImplementedError('Unknown base model {}'.format(base))
        return network
