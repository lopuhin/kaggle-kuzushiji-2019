from torch import nn


def build_model(n_classes: int):
    return Model(n_classes=n_classes)


class Model(nn.Module):
    def __init__(self, *, n_classes: int):
        super().__init__()
        n_hidden = 128
        self.fc_out = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        import IPython; IPython.embed()
