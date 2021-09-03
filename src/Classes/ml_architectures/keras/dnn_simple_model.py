import os
import numpy as np
from sklearn.model_selection import train_test_split


def run(model_obj, percentage_of_data, save):

	(x_train, y_train), (_, _) = model_obj.dataset
	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=model_obj.validation_size, random_state=model_obj.random_state)

	# for one that wants speeding up tests using part of data
	train_limit = int(len(x_train)*percentage_of_data)
	val_limit = int(len(x_valid)*percentage_of_data)
	x_train, y_train = x_train[: train_limit], y_train[: train_limit]
	x_valid, y_valid = x_valid[: val_limit], y_valid[: val_limit]

	model = model_obj.algorithm.DNN(model_obj.num_classes)
	y_train = np.eye(model_obj.num_classes)[y_train]
	y_valid = np.eye(model_obj.num_classes)[y_valid]
	model, history = model.train(x_train, y_train, x_valid, y_valid, model_obj.batch_size, model_obj.epochs)
	
	if save:
		path = os.path.join(model_obj.models_folder, model_obj.model_name)
		#saving model
		model.save(path)

	return history