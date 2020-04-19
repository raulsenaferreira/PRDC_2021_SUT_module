from src.utils import util
from src.GTRSB_experiments import DNN_GTRSB_model
from src.GTRSB_experiments import DNN_ensemble_GTRSB_model
from src.MNIST_experiments import DNN_MNIST_model


#general settings
sep = util.get_separator()
models_folder = "src"+sep+"bin"+sep+"models"+sep
validation_size = 0.3

#German traffic sign dataset
trainPath = 'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
num_classes = 43
is_classification = True
height = 28
width = 28
channels = 3

#LeNet with GTRSB dataset
#model_name = 'DNN_GTRSB.h5'
#history = DNN_GTRSB_model.run(height, width, channels, trainPath, validation_size, models_folder, 
#	model_name, is_classification, num_classes)

#Ensemble of LeNet with GTRSB dataset
#model_name_prefix = 'DNN_ensemble_GTRSB_'
#arr_history = DNN_ensemble_GTRSB_model.run(validation_size, batch_size, models_folder, epochs, model_name_prefix, sep, script_path)

#CNN with MNIST dataset
#batch_size = 128
#epochs = 12
#model_name = 'DNN_MNIST.h5'
#history = DNN_MNIST_model.run(validation_size, batch_size, models_folder, epochs, model_name)