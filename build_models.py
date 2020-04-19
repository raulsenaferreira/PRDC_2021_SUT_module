import os
from src.utils import util
from src import DNN_GTRSB_model
from src import DNN_ensemble_GTRSB_model
from src import DNN_MNIST_model
from src import SCGAN_MNIST_model


sep = util.get_separator()
script_path = os.getcwd()
epochs = 10
batch_size = 32
validation_size = 0.3
models_folder = "src"+sep+"bin"+sep+"models"+sep

#LeNet with GTRSB dataset
#model_name = 'DNN_GTRSB.h5'
#history = DNN_GTRSB_model.run(validation_size, batch_size, models_folder, epochs, model_name, sep, script_path)

#Ensemble of LeNet with GTRSB dataset
#model_name_prefix = 'DNN_ensemble_GTRSB_'
#arr_history = DNN_ensemble_GTRSB_model.run(validation_size, batch_size, models_folder, epochs, model_name_prefix, sep, script_path)

#CNN with MNIST dataset
#batch_size = 128
#epochs = 12
#model_name = 'DNN_MNIST.h5'
#history = DNN_MNIST_model.run(validation_size, batch_size, models_folder, epochs, model_name)