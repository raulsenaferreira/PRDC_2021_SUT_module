# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from scipy import misc
from torch.utils.data import TensorDataset

#from src.threats.novelty_detection.methods.sota.ODIN import threshold_metric
# my method for correct calculate the probability limits for the classes in a specific dataset
def print_thresholds(path, temperature, num_classes, magnitude):
	start = 0
	end = temperature/num_classes*magnitude
	confidence_values = np.loadtxt(path, delimiter=',')
	confidence_values = confidence_values[:, 2]
	print('For magnitude {}: min threshold {}, and max threshold {}'.format(magnitude, min(confidence_values), max(confidence_values)))


def run (monitor, net1, X, y, save, params):
	# os.getcwd(), 
	use_alternative_monitor = params['use_alternative_monitor']
	monitor.dataset_name, monitor.model_name = params['dataset_name'], params['model_name']
	use_gpu = params['use_gpu']
	temper = monitor.temperature

	X = np.moveaxis(X, -1,1)
	#print('np.shape(X)', np.shape(X))
	tensor_x = torch.Tensor(X)
	tensor_y = torch.Tensor(y)

	my_dataset = TensorDataset(tensor_x,tensor_y) 
	testloader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=0)

	if use_gpu:
		import torch.backends.cudnn as cudnn
		cudnn.benchmark = True

	CUDA_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(CUDA_DEVICE)
	net1.to(CUDA_DEVICE)

	criterion = nn.CrossEntropyLoss().to(CUDA_DEVICE)
	optimizer = optim.Adam(net1.parameters(), lr=5e-4)

	root_dir = os.path.join('src', 'threats', 'novelty_detection', 'methods', 'sota', 'ODIN', 'softmax_scores', monitor.dataset_name.lower(), monitor.model_name.lower())

	for noiseMagnitude1 in monitor.magnitude:
		path_f1 = os.path.join(root_dir, 'confidence_base_magnitude_{}.txt'.format(noiseMagnitude1))
		path_f2 = os.path.join(root_dir, 'confidence_ODIN_magnitude_{}.txt'.format(noiseMagnitude1))

		t0 = time.time()

		f1 = open(path_f1, 'w')
		g1 = open(path_f2, 'w')
		
		N = 10000
		
		for j, data in enumerate(testloader):
			
			if j<1000: continue
			images, _ = data
			
			inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
			outputs = net1(inputs)
			
			# Calculating the confidence of the output, no perturbation added here
			nnOutputs = outputs.data.cpu()
			nnOutputs = nnOutputs.numpy()
			nnOutputs = nnOutputs[0]
			nnOutputs = nnOutputs - np.max(nnOutputs)
			nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

			if save:
				f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
			
			# Using temperature scaling
			outputs = outputs / temper
			
			# Calculating the perturbation we need to add, that is,
			# the sign of gradient of cross entropy loss w.r.t. input
			maxIndexTemp = np.argmax(nnOutputs)
			labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
			loss = criterion(outputs, labels)
			loss.backward()
			
			
			# Normalizing the gradient to binary in {0, 1}
			gradient =  (torch.ge(inputs.grad.data, 0))
			gradient = (gradient.float() - 0.5) * 2
			# Normalizing the gradient to the same space of image
			gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
			gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
			gradient[0][2] = (gradient[0][2])/(66.7/255.0)
			# Adding small perturbations to images
			tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
			outputs = net1(Variable(tempInputs))
			outputs = outputs / temper
			# Calculating the confidence after adding perturbations
			nnOutputs = outputs.data.cpu()
			nnOutputs = nnOutputs.numpy()
			nnOutputs = nnOutputs[0]
			nnOutputs = nnOutputs - np.max(nnOutputs)
			nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

			if save:
				g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

			if j % 100 == 99:
				print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
				t0 = time.time()

	for noiseMagnitude1 in monitor.magnitude:
		path_f2 = os.path.join(root_dir, 'confidence_ODIN_magnitude_{}.txt'.format(noiseMagnitude1))
		
		if monitor.dataset_name.lower() == 'gtsrb':
			num_classes = 43
		elif monitor.dataset_name.lower() == 'cifar10':
			num_classes = 10

		print_thresholds(path_f2, temper, num_classes, noiseMagnitude1)