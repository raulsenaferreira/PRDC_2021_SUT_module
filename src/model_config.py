from src.Classes.model_builder import ModelBuilder
from src.Classes import dnn_simple_model
from src.Classes import cnn
from src.Classes import le_net
from src.utils import util

sep = util.get_separator()

def load_settings(model_acronym):
	model = ModelBuilder()
	model.models_folder = load_var('models_folder')

	if model_acronym == 'cnn_mnist':
		model.model_name = load_var("m1_d1_name")
		model.batch_size = load_var("m1_d1_batch")
		model.epochs = load_var("m1_d1_epoch")
		model.algorithm = cnn
		model.runner = dnn_simple_model

	elif model_acronym == 'lenet_gtsrb':
		model.model_name = load_var("m1_d2_name")
		model.batch_size = load_var("m1_d2_batch")
		model.epochs = load_var("m1_d2_epoch")
		model.algorithm = le_net
		model.runner = dnn_simple_model

	elif model_acronym == 3:
		model.model_name = load_var("m2_d1_name")
		model.batch_size = load_var("m1_d1_batch")
		model.epochs = load_var("m1_d1_epoch")
		model.runner = DNN_ensemble_MNIST_model

	elif model_acronym == 4:
		model.model_name = load_var("m2_d2_name")
		model.batch_size = load_var("m1_d2_batch")
		model.epochs = load_var("m1_d2_epoch")
		model.runner = DNN_ensemble_GTRSB_model

	return model


def load_var(key):
	var_dict = {}

	var_dict['validation_size'] = 0.3
	var_dict['models_folder'] = "src"+sep+"bin"+sep+"models"+sep
	var_dict['m1_d1_name'] = 'DNN_MNIST.h5'
	var_dict['m1_d1_batch'] = 128
	var_dict['m1_d1_epoch'] = 12

	var_dict['m1_d2_name'] = 'DNN_GTRSB.h5'
	var_dict['m1_d2_batch'] = 32
	var_dict['m1_d2_epoch'] = 10

	var_dict['m2_d1_name'] = 'DNN_ensemble_MNIST_'
	var_dict['m2_d2_name'] = 'DNN_ensemble_GTRSB_'

	return var_dict[key]