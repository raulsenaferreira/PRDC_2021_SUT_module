from src.utils import util
from src.ML_algorithms import cnn


def run(validation_size, batch_size, models_folder, epochs, model_name):
	is_classification = True
	x_train, y_train, x_test, y_test, input_shape = util.load_mnist()

	model = cnn.CNN(is_classification, num_classes, input_shape)
	model, history = model.train(x_train, y_train, x_test, y_test, epochs, batch_size)
	#saving model
	model.save(models_folder+model_name)

	return history