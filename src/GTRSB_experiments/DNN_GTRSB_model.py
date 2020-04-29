def run(num_classes, X_train, Y_train, X_valid, Y_valid, models_folder, model_obj):

	#model building and training
	DNN = le_net.LeNet(num_classes)
	model, history = DNN.train(X_train, X_valid, Y_train, Y_valid, model_obj.batch_size, model_obj.epochs)
	#saving model
	model.save(models_folder+model_obj.model_name)

	return history