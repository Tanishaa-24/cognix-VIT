"""Chest X-ray pneumonia classifier — compact custom 2D CNN, no pretrained weights needed."""

import torch.nn as nn


CLASS_NAMES = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]

ALERT_LEVEL = {
    "Normal":              "normal",
    "Bacterial Pneumonia": "high",
    "Viral Pneumonia":     "moderate",
}

CLINICAL_NOTE = {
    "Normal":              "No radiological signs of pneumonia detected.",
    "Bacterial Pneumonia": "Lobar/segmental consolidation pattern consistent with bacterial etiology. Consider antibiotic therapy.",
    "Viral Pneumonia":     "Bilateral interstitial infiltrates consistent with viral etiology. Supportive care; monitor oxygen saturation.",
}


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class XRayCNN(nn.Module):
    """
    Compact 2D CNN for 3-class chest X-ray classification.
    Input: (B, 1, 128, 128) grayscale image
    Output: (B, num_classes) logits
    Designed to train from scratch on ~800 images.
    """

    def __init__(self, num_classes: int = 3, pretrained_weights=None):
        super().__init__()
        # pretrained_weights param accepted for API compatibility but ignored
        self.features = nn.Sequential(
            ConvBlock(1, 32),    # 128→64
            ConvBlock(32, 64),   # 64→32
            ConvBlock(64, 128),  # 32→16
            ConvBlock(128, 256), # 16→8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
