from src.utils import util
from src.ML_algorithms import le_net


def run(height, width, channels, trainPath, validation_size, models_folder, model_name, is_classification, num_classes):
	epochs = 10
	batch_size = 32
	
	X_train,X_valid,Y_train,Y_valid = util.load_GTRSB_dataset(height, width, channels, trainPath, validation_size)

	#model building and training
	DNN = le_net.LeNet(is_classification, num_classes)
	model, history = DNN.train(X_train, X_valid, Y_train, Y_valid, batch_size, epochs)
	#saving model
	model.save(models_folder+model_name)

	return history