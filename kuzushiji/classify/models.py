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
        self.res_l1 = 3
        self.res_l2 = 3
        self.head = Head(
            in_features=(self.base.out_features_l1 * self.res_l1 ** 2 +
                         self.base.out_features_l2 * self.res_l2 ** 2),
            n_classes=n_classes)

    def forward(self, x):
        x, rois = x
        _, _, input_h, input_w = x.shape
        x_l1, x_l2 = self.base(x)
        del x
        x_l1 = roi_align(
            x_l1, rois,
            output_size=(self.res_l1, self.res_l1),
            spatial_scale=x_l1.shape[3] / input_w,
        )
        x_l2 = roi_align(
            x_l2, rois,
            output_size=(self.res_l2, self.res_l2),
            spatial_scale=x_l2.shape[3] / input_w,
        )
        x = torch.cat(
            [x_l1.flatten(start_dim=1),
             x_l2.flatten(start_dim=1)],
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
        self.out_features_l1 = 512
        self.out_features_l2 = 1024

    def forward(self, x):
        base = self.base
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)
        x = base.layer1(x)
        x_l1 = base.layer2(x)
        del x
        x_l2 = base.layer3(x_l1)
        return x_l1, x_l2
