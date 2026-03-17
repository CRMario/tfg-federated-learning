import torch
import torch.nn as nn
import torch.nn.functional as F

"""CNN that will be used locally:"""
class CNN_Local(nn.Module):

    def __init__(self, in_channels=3, n_labels=3):
        super(CNN_Local, self).__init__()
        out_c = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c*2)

        self.conv3 = nn.Conv2d(in_channels=out_c*2, out_channels=out_c*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c*4)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(out_c*4 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

"""CNN that will be used for BloodMNIST:"""
class CNN_BloodMNIST(nn.Module):

    def __init__(self, in_channels=3, n_labels=8):
        super(CNN_BloodMNIST, self).__init__()
        out_c = 16
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c*2)

        self.conv3 = nn.Conv2d(in_channels=out_c*2, out_channels=out_c*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c*4)

        self.pool = nn.MaxPool2d(2, 2)

        self.global_pool = nn.AdaptiveAvgPool2d((2,2))

        self.fc1 = nn.Linear(out_c*4 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

"""CNN that will be used for MNIST:"""
class CNN_MNIST(nn.Module):

    def __init__(self, input_channels=1, num_classes=10):
        super(CNN_MNIST, self).__init__()
        
        # 5x5 conv, 6 output channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)
        
        # 5x5 conv, 16 output channels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 2x2 max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
MODEL = {
    "local": CNN_Local,
    "bloodmnist": CNN_BloodMNIST,
    "mnist": CNN_MNIST,
}