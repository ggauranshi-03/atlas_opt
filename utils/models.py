import torch
import torch.nn as nn

class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(2) if pool else None
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        if self.pool: x = self.pool(x)
        return x

class AirbenchCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = ConvBN(3, 64)
        self.layer1 = ConvBN(64, 128, pool=True)
        self.layer2 = ConvBN(128, 256, pool=True)
        self.layer3 = ConvBN(256, 512, pool=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes, bias=True)
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
