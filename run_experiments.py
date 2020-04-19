import os
from src.GTRSB_experiments import DNN_outOfBox_GTRSB_test
from src.GTRSB_experiments import DNN_ensemble_outOfBox_GTRSB_test
from src.MNIST_experiments import DNN_outOfBox_MNIST_test
from src.MNIST_experiments import SCGAN_MNIST_test
import matplotlib.pyplot as plt
import numpy as np
import sys
from src.utils import util
import keras.backend as K
import scipy
from keras.losses import binary_crossentropy
from tensorflow.keras.utils import to_categorical

# ML is incorrect but monitor does not trust on it = TP
# ML is correct but monitor does not trust on it = FP
# ML is incorrect and monitor trust on it = FN
# ML is correct and monitor trust on it = TN

sep = util.get_separator()
script_path = os.getcwd()
classToMonitor = 7
isTestOneClass = True
layer_name = 'dense_1'
layer_index = 8
models_folder = "src"+sep+"bin"+sep+"models"+sep
monitors_folder = "src"+sep+"bin"+sep+"monitors"+sep
num_classes = 43
input_width = 28
input_height = 28
channels = 3
input_shape = (input_height,input_width, channels)
trainPath = 'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep

#Experiment 1: CNN with outside-of-box monitor and GTRSB dataset
#model_name = 'DNN_GTRSB.h5'
#monitor_name = "monitor_Box_GTRSB.p"
#arrPred, count, arrFP, arrFN, arrTP, arrTN = DNN_outOfBox_GTRSB_test.run(
#	classToMonitor, layer_name, models_folder, monitors_folder, monitor_name, model_name, 
#	isTestOneClass, sep, script_path)

#Experiment 2: ensemble of CNNs with outside-of-box monitor
#model_ensemble_prefix = 'DNN_ensemble_GTRSB_'
#num_cnn = 5
#monitors_ensemble_folder = "src"+sep+"bin"+sep+"monitors"+sep+"outOfBox_ensembleDNN"+sep
#monitor_ensemble_prefix = "monitor_Box_DNN_"
#arrPred, count, arrFP, arrFN, arrTP, arrTN = DNN_ensemble_outOfBox_GTRSB_test.run(classToMonitor, layer_index, 
#	models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, 
#	sep, script_path)

#Experiment 3: CNN with outside-of-box monitor and MNIST dataset
#monitor_name = "monitor_Box_MNIST.p"
#model_name = 'DNN_MNIST.h5'
#arrPred, count, arrFP, arrFN, arrTP, arrTN = DNN_outOfBox_MNIST_test.run(
#	classToMonitor, layer_name, models_folder, monitors_folder, monitor_name, model_name, 
#	isTestOneClass)

#Experiment 4 SCGAN with MNIST for novelty/OOD detection
model_name = 'DNN_MNIST.h5'
monitor_name = 'SCGAN_MNIST__3.h5'
monitors_folder = monitors_folder+'SCGAN_checkpoint'+sep
arrPred, count, arrFP, arrFN, arrTP, arrTN = SCGAN_MNIST_test.run(classToMonitor, models_folder, 
	monitors_folder, model_name, monitor_name)

#print some info about the results
util.print_positives_negatives(count, arrFP, arrFN, arrTP, arrTN, classToMonitor, isTestOneClass)