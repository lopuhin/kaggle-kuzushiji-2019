import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from torchvision import models


def build_model(base: str, n_classes: int, **kwargs) -> nn.Module:
    return Model(base=base, n_classes=n_classes, **kwargs)


class Model(nn.Module):
    def __init__(
            self, *, base: str, head: str,
            n_classes: int, head_dropout: float,
            use_sequences: bool, **base_kwargs,
            ):
        super().__init__()
        self.base = ResNetBase(base, **base_kwargs)
        self.res_l1 = 3
        self.res_l2 = 3
        self.use_sequences = use_sequences
        head_cls = globals()[head]
        self.head = head_cls(
            in_features=(self.base.out_features_l1 * self.res_l1 ** 2 +
                         self.base.out_features_l2 * self.res_l2 ** 2),
            n_classes=n_classes,
            dropout=head_dropout)
        if self.use_sequences:  # unused
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
        if self.use_sequences:  # unused
            x_features = self._apply_lstm(x_features, rois, sequences)
            x = self.head.apply_fc_out(x_features)
        return x, x_features, rois

    def _apply_lstm(self, x, rois, sequences):  # unused
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


class Head2(nn.Module):  # unused
    def __init__(self, in_features: int, n_classes: int, dropout: float):
        super().__init__()
        self.hidden_dim = 1024
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fc1 = nn.Linear(in_features, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x_features = self.bn2(x)
        x = self.apply_fc_out(x_features)
        return x, x_features

    def apply_fc_out(self, x):
        return self.fc3(x)


class Head3(nn.Module):  # unused
    def __init__(self, in_features: int, n_classes: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.apply_fc_out(x)
        return x, x

    def apply_fc_out(self, x):
        return self.fc(x)


class ResNetBase(nn.Module):
    def __init__(self, name: str, frozen_start: bool, fp16: bool):
        super().__init__()
        if name.endswith('_wsl'):
            self.base = torch.hub.load('facebookresearch/WSL-Images', name)
        else:
            self.base = getattr(models, name)(pretrained=True)
        self.frozen_start = frozen_start
        self.fp16 = fp16
        if name == 'resnet34':
            self.out_features_l1 = 256
            self.out_features_l2 = 512
        else:
            self.out_features_l1 = 512
            self.out_features_l2 = 1024

        self.frozen = []
        if self.frozen_start:
            self.frozen = [self.base.layer1, self.base.conv1, self.base.bn1]
            for m in self.frozen:
                self._freeze(m)

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

    def train(self, mode=True):
        super().train(mode=mode)
        for m in self.frozen:
            self._bn_to_eval(m)

    def _freeze(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def _bn_to_eval(self, module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
