from src.Classes.model_builder import ModelBuilder
from src.GTRSB_experiments import DNN_GTRSB_model
from src.GTRSB_experiments import DNN_ensemble_GTRSB_model
from src.MNIST_experiments import dnn_simple_model
from src.MNIST_experiments import DNN_ensemble_MNIST_model
from src.ML_algorithms import cnn
from src.ML_algorithms import le_net

def load_model_settings(model_number):
	model = ModelBuilder()

	if model_number == 1:
		model.model_name = load_var_dict("m1_d1_name")
		model.batch_size = load_var_dict("m1_d1_batch")
		model.epochs = load_var_dict("m1_d1_epoch")
		model.algorithm = cnn
		model.runner = dnn_simple_model

	elif model_number == 2:
		model.model_name = load_var_dict("m1_d2_name")
		model.batch_size = load_var_dict("m1_d2_batch")
		model.epochs = load_var_dict("m1_d2_epoch")
		model.algorithm = le_net
		model.runner = dnn_simple_model

	elif model_number == 3:
		model.model_name = load_var_dict("m2_d1_name")
		model.batch_size = load_var_dict("m1_d1_batch")
		model.epochs = load_var_dict("m1_d1_epoch")
		model.runner = DNN_ensemble_MNIST_model

	elif model_number == 4:
		model.model_name = load_var_dict("m2_d2_name")
		model.batch_size = load_var_dict("m1_d2_batch")
		model.epochs = load_var_dict("m1_d2_epoch")
		model.runner = DNN_ensemble_GTRSB_model

	return model


def load_var_dict(key):
	var_dict['models_folder'] = "src"+sep+"bin"+sep+"models"+sep
	var_dict['m1_d1_name'] = 'DNN_MNIST.h5'
	var_dict['m1_d1_batch'] = 128
	var_dict['m1_d1_epoch'] = 12
	var_dict['m1_d1_name'] = 'DNN_GTRSB.h5'
	var_dict['m1_d2_batch'] = 10
	var_dict['m1_d2_epoch'] = 32
	var_dict['m2_d1_name'] = 'DNN_ensemble_MNIST_'
	var_dict['m2_d2_name'] = 'DNN_ensemble_GTRSB_'

	return var_dict[key]