import torch
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
        self.res_l3 = 3
        self.res_l4 = 1
        self.head = Head(
            in_features=(self.base.out_features_l3 * self.res_l3 ** 2 +
                         self.base.out_features_l4 * self.res_l4 ** 2),
            n_classes=n_classes)

    def forward(self, x):
        x, rois = x
        _, _, input_h, input_w = x.shape
        x_l3, x_l4 = self.base(x)
        del x
        x_l3 = roi_align(
            x_l3, rois,
            output_size=(self.res_l3, self.res_l3),
            spatial_scale=x_l3.shape[3] / input_w,
        )
        x_l4 = roi_align(
            x_l4, rois,
            output_size=(self.res_l4, self.res_l4),
            spatial_scale=x_l4.shape[3] / input_w,
        )
        x = torch.cat(
            [x_l3.flatten(start_dim=1),
             x_l4.flatten(start_dim=1)],
            dim=1)
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
        self.out_features_l3 = 1024
        self.out_features_l4 = 2048

    def forward(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)
        x = base.layer1(x)
        x = base.layer2(x)
        x_l3 = base.layer3(x)
        del x
        x_l4 = base.layer4(x_l3)
        return x_l3, x_l4
