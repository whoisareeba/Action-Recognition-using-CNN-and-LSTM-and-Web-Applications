import torch
import torch.nn as nn
from torchvision import models


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, backbone='resnet18'):
        super(CNN_LSTM, self).__init__()

        if backbone == 'resnet34':
            self.cnn = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            cnn_out = 512
        else:
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            cnn_out = 512

        self.cnn.fc = nn.Identity()
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out
