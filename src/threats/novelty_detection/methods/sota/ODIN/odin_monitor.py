import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


def detection(net, inputs, temper, noiseMagnitude, threshold, CUDA_DEVICE, file_writter=None):
	#inputs = Variable(inputs.cuda(CUDA_DEVICE), requires_grad = True)
	inputs.requires_grad = True
	outputs = net(inputs)
	# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
	# weights ?
	nnOutputs = outputs.data.cpu()
	nnOutputs = nnOutputs.numpy()

	nnOutputs = nnOutputs[0]
	nnOutputs = nnOutputs - np.max(nnOutputs)
	# the confidence
	nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
	
	if file_writter != None:
		file_writter.write("{}, {}, {}\n".format(1, noiseMagnitude, np.max(nnOutputs)))
	
	# Using temperature scaling
	outputs = outputs / temper

	# Calculating the perturbation we need to add, that is,
	# the sign of gradient of cross entropy loss w.r.t. input
	maxIndexTemp = np.argmax(nnOutputs)
	labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))

	criterion = nn.CrossEntropyLoss()
	loss = criterion(outputs, labels)
	loss.backward()
	
	# Normalizing the gradient to binary in {0, 1}
	gradient =  torch.ge(inputs.grad.data, 0)
	gradient = (gradient.float() - 0.5) * 2

	# Normalizing the gradient to the same space of image
	gradient[0][0] = gradient[0][0]/(63.0/255.0)
	gradient[0][1] = gradient[0][1]/(62.1/255.0)
	gradient[0][2] = gradient[0][2]/(66.7/255.0)
	
	# Adding small perturbations to images
	tempInputs = torch.add(inputs.data, -noiseMagnitude, gradient)
	outputs = net(Variable(tempInputs))
	outputs = outputs / temper
	
	# Calculating the confidence after adding perturbations
	nnOutputs = outputs.data.cpu()
	nnOutputs = nnOutputs.numpy()
	nnOutputs = nnOutputs[0]
	nnOutputs = nnOutputs - np.max(nnOutputs)
	nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

	finalIndextemp = np.argmax(nnOutputs)
	#print('finalIndextemp, confidence', finalIndextemp, nnOutputs[finalIndextemp])

	# Classify it as OOD or ID
	is_OOD = True if nnOutputs[finalIndextemp] <= threshold else False

	return is_OOD