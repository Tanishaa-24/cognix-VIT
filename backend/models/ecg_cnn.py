import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.skip(x))
        return out


class ECG1DCNN(nn.Module):
    """1D ResNet for 17-class ECG arrhythmia classification."""

    CLASS_NAMES = [
        "NSR", "APB", "AFL", "AFIB", "SVTA", "WPW", "PVC",
        "Bigeminy", "Trigeminy", "VT", "IVR", "VFL", "Fusion",
        "LBBBB", "RBBBB", "SDHB", "PR",
    ]

    # Clinical cascade risk weight per class (0=low, 1=moderate, 2=high, 3=critical)
    CASCADE_RISK = {
        "NSR": 0, "APB": 1, "AFL": 2, "AFIB": 2, "SVTA": 2,
        "WPW": 2, "PVC": 1, "Bigeminy": 1, "Trigeminy": 1,
        "VT": 3, "IVR": 3, "VFL": 3, "Fusion": 1,
        "LBBBB": 1, "RBBBB": 1, "SDHB": 3, "PR": 1,
    }

    def __init__(self, num_classes=17):
        super().__init__()
        self.stem    = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1  = ResidualBlock1D(32, 32)
        self.layer2  = ResidualBlock1D(32, 64, stride=2)
        self.layer3  = ResidualBlock1D(64, 128, stride=2)
        self.layer4  = ResidualBlock1D(128, 256, stride=2)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.drop    = nn.Dropout(0.4)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)

    def get_features(self, x):
        """Returns 256-dim embedding before classifier head."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).squeeze(-1)
