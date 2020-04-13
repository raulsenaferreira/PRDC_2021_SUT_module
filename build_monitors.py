import sys
import os
from src.utils import util
from src import out_of_box_train


is_windows = sys.platform.startswith('win')
sep = '\\'
    
if is_windows == False:
    sep = '/'

script_path = os.getcwd()
classToMonitor = 7
validation_size = 0.3
K = 3
models_folder = "src"+sep+"bin"+sep+"models"+sep
model_name = 'CNN_GTRSB.h5'
layer_name = 'dense_1'
monitors_folder = "src"+sep+"bin"+sep+"monitors"+sep
monitor_name = "monitor_Box_GTRSB.p"

#building monitors
#monitoring one class in the GTRSB dataset using outside of box
out_of_box_train.run(classToMonitor, monitors_folder, monitor_name, models_folder, model_name, layer_name, validation_size, K, script_path, sep)