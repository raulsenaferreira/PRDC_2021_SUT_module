import sys
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def test(net, valLoader, PATH, device, use_gpu=True):    
    #net = Net()
    net.load_state_dict(torch.load(PATH))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in valLoader:
            if use_gpu:
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Val Accuracy: {} %'.format(100 * correct / total))


def train(net, criterion, optimizer, trainloader, num_epochs, device, path, use_gpu=True):
    #self.net = net
    
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

    print('Finishing Training...')

    torch.save(net.state_dict(), path)


def run(model_obj, percentage_of_data, save):

	(x_train, y_train), (_, _) = model_obj.dataset

	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

	dim = np.shape(x_train[0])[1]

	# for one that wants speeding up tests using part of data
	#train_limit = int(len(x_train)*percentage_of_data)
	#val_limit = int(len(x_valid)*percentage_of_data)
	#x_train, y_train = x_train[: train_limit], y_train[: train_limit]
	#x_valid, y_valid = x_valid[: val_limit], y_valid[: val_limit]

	# train
	x_train = np.moveaxis(x_train, -1,1)
	print('np.shape(x_train)', np.shape(x_train))
	tensor_x = torch.Tensor(x_train)
	tensor_y = torch.Tensor(y_train)

	my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
	trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=model_obj.batch_size,
											 shuffle=False, num_workers=0)

	
	# algorithm
	model = model_obj.algorithm.DNN(model_obj.num_classes, dim,  model_obj.batch_size)
	path = os.path.join(os.getcwd(), model_obj.models_folder, model_obj.model_name)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	model.to(device)
	cudnn.benchmark = True

	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = optim.Adam(model.parameters(), lr=5e-4)#SGD(net.parameters(), lr=0.001, momentum=0.9)

	#y_train = np.eye(model_obj.num_classes)[y_train]
	#y_valid = np.eye(model_obj.num_classes)[y_valid]
	train(model, criterion, optimizer, trainloader, model_obj.epochs, device, path)
	
	#test
	x_valid = np.moveaxis(x_valid, -1,1)
	print('np.shape(x_valid)', np.shape(x_valid))
	tensor_x = torch.Tensor(x_valid)
	tensor_y = torch.Tensor(y_valid)

	my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
	valLoader = torch.utils.data.DataLoader(my_dataset, batch_size=1,
											 shuffle=False, num_workers=0)

	test(model, valLoader, path, device)

	return True