def run(model_obj, percentage_of_data):

	x_train, y_train, x_valid, y_valid = model_obj.dataset.load_dataset(mode='train')
	# for one that wants speeding up tests using part of data
	train_limit = int(len(x_train)*percentage_of_data)
	val_limit = int(len(x_valid)*percentage_of_data)
	x_train, y_train = x_train[: train_limit], y_train[: train_limit]
	x_valid, y_valid = x_valid[: val_limit], y_valid[: val_limit]

	model = model_obj.algorithm.DNN(model_obj.dataset.num_classes)
	model, history = model.train(x_train, y_train, x_valid, y_valid, model_obj.batch_size, model_obj.epochs)
	#saving model
	model.save(model_obj.models_folder+model_obj.model_name)

	return history