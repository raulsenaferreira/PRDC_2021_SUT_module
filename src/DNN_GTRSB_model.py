from src.utils import util
from src.ML_algorithms import le_net


def run(validation_size, batch_size, models_folder, epochs, model_name, sep, script_path):
	is_classification = True
	num_classes = 43
	
	#loading German traffic sign dataset
	trainPath = str(script_path)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
	X_train,X_valid,Y_train,Y_valid = util.load_GTRSB_dataset(trainPath, validation_size)

	#model building and training
	DNN = le_net.LeNet(is_classification, num_classes)
	model, history = DNN.train(X_train, X_valid, Y_train, Y_valid, batch_size, epochs)
	#saving model
	model.save(models_folder+model_name)

	return history