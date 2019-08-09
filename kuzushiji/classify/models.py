from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from torchvision import models


def build_model(n_classes: int) -> nn.Module:
    return Model(n_classes=n_classes)


class Model(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.base = ResNetBase()
        self.resolution = 3
        self.head = Head(
            in_features=self.base.out_features * self.resolution ** 2,
            n_classes=n_classes)

    def forward(self, x, rois):
        x = self.base(x)
        x = roi_align(x, rois, output_size=self.resolution)
        # TODO flatten or reshape
        x = self.head(x)
        return x


class Head(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        hidden_dim = 1024
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x


class ResNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(pretrained=True)
        self.out_features = self.base.fc.in_features

    def forward(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)
        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)
        return x
