import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=3, dropout=0.5, feature_dim=128):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lstm = nn.LSTM(feature_dim, feature_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1)  # (B, T, feature_dim)
        feats, _ = self.lstm(feats)
        feats = feats[:, -1, :]  # last time step
        feats = self.dropout(feats)
        logits = self.classifier(feats)
        return logits.squeeze(1)
