import sys
import os
from src.utils import util
from src import out_of_box_test_GTRSB
#from src import ensembleDNN_outOfBox_test


is_windows = sys.platform.startswith('win')
sep = '\\'
    
if is_windows == False:
    sep = '/'

script_path = os.getcwd()
classToMonitor = 7
isTestOneClass = True
layer_name = 'dense_1'
models_folder = "src"+sep+"bin"+sep+"models"+sep
monitors_folder = "src"+sep+"bin"+sep+"monitors"+sep

#run experiment
count, arrFP, arrFN, arrTP, arrTN = out_of_box_test_GTRSB.run(
	classToMonitor, layer_name, models_folder, monitors_folder, isTestOneClass, sep, script_path)

#print some info about the results
util.print_positives_negatives(count, arrFP, arrFN, arrTP, arrTN, classToMonitor, isTestOneClass)