import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 16),
            self._block(16, 32),
            self._block(32, 64),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, class_count),
        )

    def _block(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, image):
        return self.classifier(self.features(image))
