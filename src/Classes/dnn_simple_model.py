def run(model_obj):

	x_train, y_train, x_valid, y_valid = model_obj.dataset.load_dataset(mode='train')
	model = model_obj.algorithm.DNN(model_obj.dataset.num_classes)
	model, history = model.train(x_train, y_train, x_valid, y_valid, model_obj.batch_size, model_obj.epochs)
	#saving model
	model.save(model_obj.models_folder+model_obj.model_name)

	return history