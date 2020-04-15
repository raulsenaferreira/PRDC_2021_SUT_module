import sys
import os
from src.utils import util
from src import CNN_GTRSB
from src import CNN_ensemble_GTRSB


is_windows = sys.platform.startswith('win')
sep = '\\'
    
if is_windows == False:
    sep = '/'

script_path = os.getcwd()
epochs = 10
batch_size = 32
validation_size = 0.3
models_folder = "src"+sep+"bin"+sep+"models"+sep
model_name = 'CNN_GTRSB.h5'
model_name_prefix = 'CNN_ensemble_GTRSB_'

#CNN with GTRSB dataset
#history = CNN_GTRSB.run(validation_size, batch_size, models_folder, epochs, model_name, sep, script_path)

#Ensemble of CNNs with GTRSB dataset
arr_history = CNN_ensemble_GTRSB.run(validation_size, batch_size, models_folder, epochs, model_name_prefix, sep, script_path)