from src.utils import util
from src.ML_algorithms import cnn


def run(validation_size, batch_size, models_folder, epochs, model_name):
	is_classification = True
	num_classes = 10
	x_train, y_train, x_valid, y_valid, x_test, y_test, input_shape = util.load_mnist()

	model = cnn.CNN(is_classification, num_classes, input_shape)
	model, history = model.train(x_train, y_train, x_valid, y_valid, epochs, batch_size)
	#saving model
	model.save(models_folder+model_name)

	return history