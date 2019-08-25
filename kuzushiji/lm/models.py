from torch import nn


def build_model(n_classes: int):
    return Model(n_classes=n_classes)


class Model(nn.Module):
    def __init__(
            self, *,
            n_classes: int,
            embedding_dim: int = 256,
            hidden_dim: int = 256,
            ):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x.transpose(1, 2)
