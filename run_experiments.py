import sys
import os
from src.utils import util
from src import out_of_box_test_GTRSB
#from src import ensembleDNN_outOfBox_test_GTRSB


# ML is incorrect but monitor does not trust on it = TP
# ML is correct but monitor does not trust on it = FP
# ML is incorrect and monitor trust on it = FN
# ML is correct and monitor trust on it = TN

is_windows = sys.platform.startswith('win')
sep = '\\'
    
if is_windows == False:
    sep = '/'

script_path = os.getcwd()
classToMonitor = 7
isTestOneClass = True
layer_name = 'dense_1'
layer_index = 8
models_folder = "src"+sep+"bin"+sep+"models"+sep
monitors_folder = "src"+sep+"bin"+sep+"monitors"+sep

#specific for experiment 1
model_name = 'CNN_GTRSB.h5'
monitor_name = "monitor_Box_GTRSB.p"

#specific for experiment 2
model_ensemble_prefix = 'CNN_ensemble_GTRSB_'
num_cnn = 5
monitors_ensemble_folder = "src"+sep+"bin"+sep+"monitors"+sep+"outOfBox_ensembleDNN"+sep
monitor_ensemble_prefix = "monitor_Box_DNN_"

#Experiment 1: CNN with outside-of-box monitor
arrPred, count, arrFP, arrFN, arrTP, arrTN = out_of_box_test_GTRSB.run(
	classToMonitor, layer_name, models_folder, monitors_folder, monitor_name, model_name, 
	isTestOneClass, sep, script_path)

#Experiment 2: ensemble of CNNs with outside-of-box monitor
arrPred, count, arrFP, arrFN, arrTP, arrTN = ensembleDNN_outOfBox_test_GTRSB.run(classToMonitor, layer_index, 
	models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, 
	sep, script_path)

#print some info about the results
util.print_positives_negatives(count, arrFP, arrFN, arrTP, arrTN, classToMonitor, isTestOneClass)