import torch
import torch.nn as nn
import torch.nn.functional as F



class DNN(nn.Module):
    def __init__(self, num_classes, dim, batch_size):
        super(DNN, self).__init__()
        
        if dim == 28:
            width_height = 4
        elif dim == 32:
            width_height = 5

        self.batch_size = batch_size
        self.conv_channels = 16
        self.fc1_input = self.conv_channels * width_height * width_height

        self.num_classes = num_classes
        self.net = None
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.conv_channels, kernel_size=5)
        self.fc1 = nn.Linear(self.fc1_input, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes) # 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), self.fc1_input)
        #x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x