
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolution: 1 input channel, 8 output channels
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # Activation function
        self.relu = nn.ReLU()

        # Max pooling
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully connected layer
        self.fc = nn.Linear(
            in_features=8 * 13 * 13,
            out_features=10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return x
