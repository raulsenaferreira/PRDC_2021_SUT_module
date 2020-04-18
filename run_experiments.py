import os
from src import DNN_outOfBox_GTRSB_test
from src import DNN_ensemble_outOfBox_GTRSB_test
from src import DNN_outOfBox_MNIST_test
from src.ML_algorithms import SCGAN_MNIST_2
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
#specific for experiment 1
model_name = 'DNN_GTRSB.h5'
monitor_name = "monitor_Box_GTRSB.p"

#specific for experiment 2
model_ensemble_prefix = 'DNN_ensemble_GTRSB_'
num_cnn = 5
monitors_ensemble_folder = "src"+sep+"bin"+sep+"monitors"+sep+"outOfBox_ensembleDNN"+sep
monitor_ensemble_prefix = "monitor_Box_DNN_"
trainPath = 'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep

#specific for experiment 3
output_height = 28
output_width = 28

#Experiment 1: CNN with outside-of-box monitor and GTRSB dataset
#arrPred, count, arrFP, arrFN, arrTP, arrTN = DNN_outOfBox_GTRSB_test.run(
#	classToMonitor, layer_name, models_folder, monitors_folder, monitor_name, model_name, 
#	isTestOneClass, sep, script_path)

#Experiment 2: ensemble of CNNs with outside-of-box monitor
#arrPred, count, arrFP, arrFN, arrTP, arrTN = DNN_ensemble_outOfBox_GTRSB_test.run(classToMonitor, layer_index, 
#	models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, 
#	sep, script_path)

#Experiment 3: CNN with outside-of-box monitor and MNIST dataset
monitor_name = "monitor_Box_MNIST.p"
model_name = 'DNN_MNIST.h5'
arrPred, count, arrFP, arrFN, arrTP, arrTN = DNN_outOfBox_MNIST_test.run(
	classToMonitor, layer_name, models_folder, monitors_folder, monitor_name, model_name, 
	isTestOneClass)

#print some info about the results
util.print_positives_negatives(count, arrFP, arrFN, arrTP, arrTN, classToMonitor, isTestOneClass)

#dcgan = GAN.DCGAN(trainPath)
#dcgan.train(epochs=4000, batch_size=32, save_interval=500)

#Experiment 4 SCGAN with MNIST for novelty/OOD detection
#model = SCGAN_MNIST_2.ALOCC_Model(dataset_name='mnist', input_height=28,input_width=28)
#model.train(epochs=5, batch_size=128, sample_interval=500)