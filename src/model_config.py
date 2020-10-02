import os
from src.Classes.model_builder import ModelBuilder
from src.Classes.ml_architectures import dnn_simple_model
from src.Classes.ml_architectures import cnn
from src.Classes.ml_architectures import le_net
from src.Classes.ml_architectures import resnet
from src.Classes.ml_architectures import vgg16



def load_settings(model_acronym):
	model = ModelBuilder()
	model.models_folder = os.path.join("src", "bin", "models")
	model.validation_size = 0.3

	if model_acronym == 'lenet_mnist':
		model.model_name = 'leNet_MNIST.h5'
		model.batch_size = 128
		model.epochs = 12
		model.algorithm = cnn
		model.runner = dnn_simple_model

	elif model_acronym == 'lenet_gtsrb':
		model.model_name = 'leNet_GTSRB.h5'
		model.batch_size = 32
		model.epochs = 10
		model.algorithm = le_net
		model.runner = dnn_simple_model

	elif model_acronym == 'resnet_cifar10':
		model.model_name = 'resNet_CIFAR-10.h5'
		model.batch_size = 32
		model.epochs = 200
		model.algorithm = resnet
		model.runner = dnn_simple_model

	elif model_acronym == 'vgg16_gtsrb':
		model.model_name = 'vgg16_GTSRB.h5'
		model.batch_size = 100
		model.epochs = 10
		model.algorithm = vgg16
		model.runner = dnn_simple_model

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

	return model