import numpy as np
from src.utils import util
from src.ML_algorithms import le_net


def run(validation_size, batch_size, models_folder, epochs, model_name_prefix, sep, script_path):
	is_classification = True
	num_classes = 43
	arr_history = []
	
	#loading German traffic sign dataset
	trainPath = str(script_path)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
	X_train,X_valid,Y_train,Y_valid = util.load_GTRSB_dataset(trainPath, validation_size)

	#image pre-processing functions
	array_function=[util.image_adjustment, util.histogram_equalization, util.adaptive_hist_eq, util.contrast_normalization]

	for i in range(len(array_function)+1):

		if i < len(array_function):
			X_train = np.array(list(map(array_function[i], X_train)))
			X_valid = np.array(list(map(array_function[i], X_valid)))

		#model building and training
		DNN = le_net.LeNet(is_classification, num_classes)
		model, history = DNN.train(X_train, X_valid, Y_train, Y_valid, batch_size, epochs)
		arr_history.append(history)
		
		#saving model
		model.save(models_folder+model_name_prefix+str(i)+'.h5')

	return arr_history