import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        # 第一个卷积层和下采样
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.pool1 = nn.AvgPool2d(2)
        # 第二个卷积层和下采样
        self.conv2 = nn.Conv2d(6, 16, 5,1)
        self.pool2 = nn.AvgPool2d(2)
        # 第三个卷积层
        self.conv3 = nn.Conv2d(16, 120, 5, 1)
        # 全连接层
        self.f1 = nn.Linear(120, 84)
        self.f2 = nn.Linear(84, n_classes)
        # 激活函数
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = torch.flatten(x, 1)

        x = self.f1(x)
        x = self.activation(x)
        logits = self.f2(x)
        probs = F.softmax(logits, dim=1)
        
        return logits, probs
