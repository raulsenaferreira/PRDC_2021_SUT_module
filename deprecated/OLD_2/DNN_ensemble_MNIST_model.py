import numpy as np
from src.utils import util
from src.ML_algorithms import cnn


def run(batch_size, models_folder, epochs, model_name_prefix):
	is_classification = True
	num_classes = 10
	arr_history = []
	
	x_train, y_train, x_valid, y_valid, _, _, input_shape = util.load_mnist()

	#image pre-processing functions
	array_function=[util.std_normalization]
	len_arr = len(array_function)
	array_function_2=[util.image_adjustment]

	for i in range(len_arr):
		model = cnn.CNN(is_classification, num_classes, input_shape)
		
		train_iterator = array_function[i](x_train, y_train, model, epochs, batch_size)
		#model training
		model, history = model.train(x_train, y_train, x_valid, y_valid, epochs, batch_size, True, train_iterator)
		arr_history.append(history)
		
		#saving model
		model.save(models_folder+model_name_prefix+str(i)+'.h5')

	for i in range(len(array_function_2)+1):
		if i < len(array_function_2):
			x_train = np.array(list(map(array_function_2[i], x_train)))

			#model building and training
		model = cnn.CNN(is_classification, num_classes, input_shape)
		model, history = model.train(x_train, y_train, x_valid, y_valid, epochs, batch_size)
		arr_history.append(history)
		
		#saving model
		model.save(models_folder+model_name_prefix+str(i+len_arr)+'.h5')
	
	return arr_history