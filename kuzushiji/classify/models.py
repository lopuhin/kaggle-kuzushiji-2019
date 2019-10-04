import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from torchvision import models


def build_model(base: str, n_classes: int, **kwargs) -> nn.Module:
    return Model(base=base, n_classes=n_classes, **kwargs)


class Model(nn.Module):
    def __init__(
            self, base: str, n_classes: int, head_dropout: float,
            use_sequences: bool,
            ):
        super().__init__()
        self.base = ResNetBase(base)
        self.res_l1 = 3
        self.res_l2 = 3
        self.use_sequences = use_sequences
        self.head = Head(
            in_features=(self.base.out_features_l1 * self.res_l1 ** 2 +
                         self.base.out_features_l2 * self.res_l2 ** 2),
            n_classes=n_classes,
            dropout=head_dropout)
        if self.use_sequences:
            self.lstm = nn.LSTM(
                input_size=self.head.hidden_dim,
                hidden_size=self.head.hidden_dim // 2,
                bidirectional=True)

    def forward(self, x):
        x, rois, sequences = x
        _, _, input_h, input_w = x.shape
        x_l1, x_l2 = self.base(x)
        dtype = x_l1.dtype
        rois = [roi.to(dtype) for roi in rois]
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
        x, x_features = self.head(x)
        if self.use_sequences:
            x_features = self._apply_lstm(x_features, rois, sequences)
            x = self.head.apply_fc_out(x_features)
        return x, x_features, rois

    def _apply_lstm(self, x, rois, sequences):
        assert len(rois) == len(sequences)
        assert x.shape[0] == sum(map(len, rois))
        offset = 0
        output = torch.zeros_like(x)
        for item_rois, item_sequences in zip(rois, sequences):
            assert item_rois.shape[0] == sum(map(len, item_sequences))
            for sequence in item_sequences:
                offset_sequence = sequence + offset
                seq_input = x[offset_sequence]
                seq_output, _ = self.lstm(seq_input.unsqueeze(1))
                output[offset_sequence] = seq_output.squeeze(1)
            offset += item_rois.shape[0]
        return output


def get_output(x_rois):
    x, x_features, rois = x_rois
    return x


class Head(nn.Module):
    def __init__(self, in_features: int, n_classes: int, dropout: float):
        super().__init__()
        self.hidden_dim = 1024
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fc1 = nn.Linear(in_features, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x_features = self.bn(x)
        x = self.apply_fc_out(x_features)
        return x, x_features

    def apply_fc_out(self, x):
        return self.fc2(x)


class ResNetBase(nn.Module):
    def __init__(self, name: str = 'resnet50'):
        super().__init__()
        if name.endswith('_wsl'):
            self.base = torch.hub.load('facebookresearch/WSL-Images', name)
        else:
            self.base = getattr(models, name)(pretrained=True)
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
