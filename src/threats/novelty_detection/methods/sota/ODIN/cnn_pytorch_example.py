import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from classes import classifiers
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
import odin_monitor
import numpy as np
import os
import numpy as np
from PIL import Image
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset
import math
'''
Credits to Stanislav Arnaudov 
https://palikar.github.io/posts/pytorch_datasplit/
'''

import logging
from functools import lru_cache
from torch.utils.data.sampler import SubsetRandomSampler



class DataSplit:

    def __init__(self, dataset, test_train_split=0.7, val_train_split=0.2, shuffle=False):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[ : validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader

class Dataset:
    """docstring for Dataset"""
    def __init__(self, root_dir='data', transform=None):
        super(Dataset, self).__init__()
        self.dataset_name = ''
        self.modification = ''
        self.dataset_ID_name = ''
        self.dataset_OOD_name = ''
        self.root_dir = root_dir
        self.width = 0
        self.height = 0
        self.channels = 0
        self.testPath = ''
        self.num_classes = 0
        self.trainPath = ''
        self.testPath = ''
        self.validation_size = None
        self.X = [[]]
        self.y = []
        self.transform = transform


    def __getitem__(self, index):
        image = self.X[index]
        
        # Just apply your transformations here
        if self.transform is not None:
            image = self.transform(image)
        x = TF.to_tensor(image)
        
        return x, self.y[index]


    def __len__(self):
        return len(self.y)

    
    def load_dataset(self, dataset_path, mode='train'):
        x_train, y_train, x_test, y_test = None, None, None, None
        
        train_images = os.path.join(dataset_path, 'train-images-npy.gz')
        train_labels = os.path.join(dataset_path, 'train-labels-npy.gz')
    
        test_images = os.path.join(dataset_path, 'test-images-npy.gz')
        test_labels = os.path.join(dataset_path, 'test-labels-npy.gz')

        if mode == 'train' or mode == 'all':
            f = gzip.GzipFile(train_images, "r")
            x_train = np.load(f)
        
            f = gzip.GzipFile(train_labels, "r")
            y_train = np.load(f)

        elif mode == 'test' or mode == 'all':
            f = gzip.GzipFile(test_images, "r")
            x_test = np.load(f)

            f = gzip.GzipFile(test_labels, "r")
            y_test = np.load(f)

        return (x_train, y_train), (x_test, y_test)



class LeNet(nn.Module):
    def __init__(self, num_classes, batch_size):
        super(LeNet, self).__init__()
        self.batch_size = batch_size
        self.conv_channels = 16
        self.fc1_input = self.conv_channels * 4 * 4

        #self.height = math.sqrt( self.fc1_input / self.batch_size / self.conv_channels)
        #self.width = math.sqrt( self.fc1_input / self.batch_size / self.conv_channels)

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, self.conv_channels, kernel_size=5)
        self.fc1 = nn.Linear(self.fc1_input, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), self.fc1_input)
        #x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

root_dir = os.path.join('C:', '\\Users', 'rsenaferre', 'Desktop', 'GITHUB', 'phd_data_generation', 'data', 'training_set')
dataset_name = 'gtsrb'
dataset_path = os.path.join(root_dir, dataset_name)
use_gpu = True

magnitude = 0.0014 # epsilon no paper
threshold = 0.1007
temperature = 1000
batch_size = 100

# lenet
net = LeNet(43, batch_size)
#PATH = 'models/lenet/cifar10.pth'
PATH = 'models/lenet/gtsrb.pth'
num_epochs = 30

# vgg 16
#net = classifiers.CIFAR10_VGG()
#PATH = 'models/vgg/cifar10.pth'

if use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    cudnn.benchmark = True

dataset = Dataset()
(x_train, y_train), (_, _) = dataset.load_dataset(dataset_path)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=42)


# train
x_train = np.moveaxis(x_train, -1,1)
print('np.shape(x_train)', np.shape(x_train))
tensor_x = torch.Tensor(x_train)
tensor_y = torch.Tensor(y_train)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

#test
x_valid = np.moveaxis(x_valid, -1,1)
print('np.shape(x_valid)', np.shape(x_valid))
tensor_x = torch.Tensor(x_valid)
tensor_y = torch.Tensor(y_valid)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
testloader = torch.utils.data.DataLoader(my_dataset, batch_size=1,
                                         shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=5e-4)#SGD(net.parameters(), lr=0.001, momentum=0.9)


def train(trainloader, num_epochs=40):

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if use_gpu:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        #test(valLoader)

    print('Finished Training')

    torch.save(net.state_dict(), PATH)


def test(testloader):    
    #net = Net()
    net.load_state_dict(torch.load(PATH))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            if use_gpu:
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Val Accuracy: {} %'.format(100 * correct / total))


def test_ODIN():    
    #net = Net()
    net.load_state_dict(torch.load(PATH))
    net.eval()

    correct = 0
    total = 0
    OOD_detection = 0
    total_OOD = 0
    with torch.no_grad():
        for data in testloader:
            if use_gpu:
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data
            print(np.shape(images))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # odin detector
            nnOutputs, is_OOD = odin_monitor.detection(net, images, temperature, magnitude,
              threshold, device)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            OOD_detection += 1 if is_OOD else 0

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    print('False positives:',OOD_detection)

if __name__ == '__main__':
    # lenet cifar (0.64 accuracy) num_epochs=40, batch size=100, lr=lr=5e-4
    # lenet gtsrb (0.97 accuracy) num_epochs=30, batch size=100, lr=lr=5e-4
    train(trainloader, num_epochs)
    test(testloader)
    #test_ODIN()