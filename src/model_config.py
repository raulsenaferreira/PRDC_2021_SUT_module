import os
from src.Classes.model_builder import ModelBuilder



def load_settings(name, dataset, backend):
	model_acronym = '{}_{}'.format(name, dataset)

	model = ModelBuilder() 
	model.models_folder = os.path.join("src", "bin", "models", backend)
	model.validation_size = 0.3
	model.random_state = 42

	if backend == 'keras':
		from src.Classes.ml_architectures.keras import dnn_simple_model
		model.runner = dnn_simple_model

		if model_acronym == 'lenet_mnist':
			model.num_classes = 10
			from src.Classes.ml_architectures.keras import cnn
			model.model_name = 'leNet_MNIST.h5'
			model.batch_size = 128
			model.epochs = 12
			model.algorithm = cnn

		elif model_acronym == 'lenet_gtsrb':
			model.num_classes = 43
			from src.Classes.ml_architectures.keras import le_net
			model.model_name = 'leNet_GTSRB.h5'
			model.batch_size = 32
			model.epochs = 10
			model.algorithm = le_net

		elif model_acronym == 'resnet_cifar10':
			model.num_classes = 10
			from src.Classes.ml_architectures.keras import resnet
			model.model_name = 'resNet_CIFAR-10.h5'
			model.batch_size = 32
			model.epochs = 200
			model.algorithm = resnet

		elif model_acronym == 'vgg16_gtsrb':
			from src.Classes.ml_architectures.keras import vgg16
			model.model_name = 'vgg16_GTSRB.h5'
			model.batch_size = 100
			model.epochs = 10
			model.algorithm = vgg16
		'''
		elif model_acronym == 3:
			model.model_name = 'DNN_ensemble_MNIST_'
			model.batch_size = 128
			model.epochs = 12
			model.runner = DNN_ensemble_MNIST_model

		elif model_acronym == 4:
			model.model_name = 'DNN_ensemble_GTRSB_'
			model.batch_size = 32
			model.epochs = 10
			model.runner = DNN_ensemble_GTRSB_model
		'''
	elif backend == 'pytorch':
		from src.Classes.ml_architectures.pytorch import dnn_simple_model
		model.runner = dnn_simple_model

		if model_acronym == 'lenet_cifar10':
			from src.Classes.ml_architectures.pytorch import pytorch_classifiers
			model.model_name = 'leNet_CIFAR-10.pth'
			model.batch_size = 100
			model.epochs = 40
			model.algorithm = pytorch_classifiers
			model.num_classes = 10

		elif model_acronym == 'lenet_gtsrb':
			from src.Classes.ml_architectures.pytorch import pytorch_classifiers
			model.model_name = 'leNet_GTSRB.pth'
			model.batch_size = 100
			model.epochs = 30
			model.algorithm = pytorch_classifiers
			model.num_classes = 43

	return model