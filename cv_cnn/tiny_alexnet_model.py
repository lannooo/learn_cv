import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniAlexNet(nn.Module):
    def __init__(self, n_classes=10, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            nn.Conv2d(16, 48, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Linear(64*4*4, 1024), 
            nn.ReLU(inplace=True), 
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = self.features(x) # (batch, 64, 4, 4)
        x = self.avgpool(x) # ajust different image shape
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        prob = F.softmax(logits, dim=1)
        return logits, prob


if __name__ == '__main__':
    net = MiniAlexNet()
    x = torch.randn((5, 3, 32, 32))
    logits, probs = net(x)
    print(logits.shape)
    print(probs.shape)