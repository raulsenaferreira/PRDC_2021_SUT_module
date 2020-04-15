import sys
import os
from src.utils import util
from src import DNN_outOfBox_GTRSB_monitor
from src import DNN_ensemble_outOfBox_GTRSB_monitor


is_windows = sys.platform.startswith('win')
sep = '\\'
    
if is_windows == False:
    sep = '/'

script_path = os.getcwd()
classToMonitor = 7
validation_size = 0.3
K = 3
num_cnn = 5
models_folder = "src"+sep+"bin"+sep+"models"+sep
model_name = 'DNN_GTRSB.h5'
layer_name = 'dense_1'
layer_index = 8
monitors_folder = "src"+sep+"bin"+sep+"monitors"+sep
monitor_name = "monitor_Box_GTRSB.p"
monitors_ensemble_folder = monitors_folder+"outOfBox_ensembleDNN"+sep
monitor_ensemble_prefix = "monitor_Box_DNN_"
model_ensemble_prefix = 'DNN_ensemble_GTRSB_'

#monitoring one class in the GTRSB dataset using outside of box
#success = DNN_outOfBox_GTRSB_monitor.run(classToMonitor, monitors_folder, monitor_name, models_folder, model_name, layer_name, validation_size, K, script_path, sep)

#monitoring ensemble of CNNs
#success = DNN_ensemble_outOfBox_GTRSB_monitor.run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, validation_size, K, sep, script_path)