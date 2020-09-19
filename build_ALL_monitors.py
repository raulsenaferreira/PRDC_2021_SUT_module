import os
import logging
import numpy as np
from src.utils import util
from src import model_config as model_cfg
from src.Classes.model_builder import ModelBuilder
from src.novelty_detection import config as config_ND
from pathos.multiprocessing import ProcessingPool as Pool
from src.Classes.dataset import Dataset
from src.novelty_detection import build_monitors
from timeit import default_timer as timer
from keras.models import load_model



def set_tf_loglevel(level):
	if level >= logging.FATAL:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	if level >= logging.ERROR:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	if level >= logging.WARNING:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
	else:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
	logging.getLogger('tensorflow').setLevel(level)


def build_monitors_by_class(root_path, parallel_execution, dataset_name, params, classes_to_monitor, model_file, X, y, save):
	arr_monitors = []

	# Generate monitors for each class for a specific dataset
	for class_to_monitor in range(classes_to_monitor):
		#Building monitors for Novelty Detection
		monitors = build_monitors.prepare_box_based_monitors(root_path, dataset_name, params, class_to_monitor)
		arr_monitors.extend(monitors)
		#arr_monitors = np.append(arr_monitors, monitors)
	
	#Parallelizing the experiments (optional): one experiment per Core
	if parallel_execution:
		cores = 6
		timeout = 30 * len(arr_monitors)
		pool = Pool(cores)
		processes_pool = []

		print("\nParallel execution with {} cores. Max {} seconds to run each experiment:".format(cores, timeout))

		for monitor in arr_monitors:
			processes_pool.append(pool.apipe(monitor.trainer.run, monitor, model_file, X, y, save)) 
		
		for process in processes_pool:
			_, _ = process.get(timeout=timeout)
	else:
		print("\nSerial execution.")
		for monitor in arr_monitors:
			_, _ = monitor.trainer.run(monitor, model_file, X, y, save)
			

def build_monitors_all_classes(root_path, parallel_execution, dataset_name, params, model_file, X, y, save):
	# Generate monitors for all classes for a specific dataset
	arr_monitors = build_monitors.prepare_monitors(root_path, dataset_name, params)
	
	#Parallelizing the experiments (optional): one experiment per Core
	if parallel_execution:
		cores = 6
		timeout = 30 * len(arr_monitors)
		pool = Pool(cores)
		processes_pool = []

		print("\nParallel execution with {} cores. Max {} seconds to run each experiment:".format(cores, timeout))

		for monitor in arr_monitors:
			processes_pool.append(pool.apipe(monitor.trainer.run, monitor, model_file, X, y, save)) 
		
		for process in processes_pool:
			process.get(timeout=timeout)
	else:
		print("\nSerial execution.")
		for monitor in arr_monitors:
			monitor.trainer.run(monitor, model_file, X, y, save)



if __name__ == "__main__":
	# disabling tensorflow logs
	set_tf_loglevel(logging.FATAL)
	# re-enabling tensorflow logs
	#set_tf_loglevel(logging.INFO)

	#general settings
	sep = util.get_separator()
	save = True
	parallel_execution = False

	sub_field = 'novelty_detection'
	dataset_names = ['GTSRB']# 'MNIST',
	validation_size = 0.3
	model_names = ['leNet'] #, 'leNet'
	
	PARAMS = {
	 #for oob variations
	 'arr_n_components' : [2], #2, 3, 5, 10
	 #for oob variations and knn
	 'arr_n_clusters' : [2, 3, 5, 10], #1, 2, 3, 4, 5
	 #for hdbscan
	 'min_samples': [5, 10, 15],  #min_samples 5, 10, 15
	 #for random forest and linear classifiers
	 'use_grid_search' : True, 
	 #for knn and sgd classifiers
	 'use_scaler': True,
	 #all methods
	 'use_alternative_monitor': False, # True = label -> act func; False = label -> act func if label == predicted
	 'technique_names' : ['sgd']} #'baseline', 'knn', 'random_forest', 'sgd', 'hdbscan', 'oob', 'oob_isomap', 'oob_pca', 'oob_pca_isomap'

	num_classes_to_monitor = [43]# 10, 43
	is_build_monitors_by_class = False
	perc_of_data = 1 #e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data

	root_path = 'src'+sep+sub_field+sep+'bin'+sep+'monitors'
	
	for model_name, dataset_name, classes_to_monitor in zip(model_names, dataset_names, num_classes_to_monitor):
		
		#path to load the model
		models_folder = "src"+sep+"bin"+sep+"models"+sep
		model_file = models_folder+model_name+'_'+dataset_name+'.h5'
		# loading model
		model = load_model(model_file)

		# loading dataset
		dataset = Dataset(dataset_name)
		dataset.validation_size = validation_size
		
		x_train, y_train, x_valid, y_valid = dataset.load_dataset(mode='train')
		x_train, y_train = x_train[:int(len(x_train)*perc_of_data)], y_train[:int(len(y_train)*perc_of_data)]
		x_valid, y_valid = x_valid[:int(len(x_valid)*perc_of_data)], y_valid[:int(len(y_valid)*perc_of_data)]

		#building monitor with training + validation data (test data is excluded, of course)
		X = np.concatenate([x_train, x_valid], axis=0)	
		y = np.vstack((y_train, y_valid))

		start = timer()

		if is_build_monitors_by_class:
			build_monitors_by_class(root_path, parallel_execution, dataset_name, PARAMS, classes_to_monitor, model, X, y, save)
		else:
			build_monitors_all_classes(root_path, parallel_execution, dataset_name, PARAMS, model, X, y, save)
		
		dt = timer() - start
		print("Monitors for {} built in {} minutes".format(dataset_name, dt/60))
	

	
	
#monitoring ensemble of CNNs in the GTRSB using outside of box
#layer_index = 8
#monitors_ensemble_folder = monitors_folder+"outOfBox_ensembleDNN"+sep
#monitor_ensemble_prefix = "monitor_Box_DNN_"
#model_ensemble_prefix = 'DNN_ensemble_GTRSB_'
#num_cnn = 5
#success = DNN_ensemble_outOfBox_GTRSB_monitor.run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, validation_size, K, sep, script_path)

#monitoring one class in the MNIST dataset using outside of box
#monitor_name = "monitor_Box_MNIST.p"
#model_name = 'DNN_MNIST.h5'
#success = DNN_outOfBox_MNIST_monitor.run(classToMonitor, monitors_folder, monitor_name, models_folder, model_name, layer_name, validation_size, K)

#monitoring one classe in the MNIST dataset using a DCGAN:
#epochs=5
#batch_size=128
#sample_interval=500
#monitors_folder_checkpoint = monitors_folder+sep+'SCGAN_checkpoint'
#monitor_name = 'SCGAN_MNIST_'
#monitor = SCGAN_MNIST_monitor.ALOCC_Model(input_height=28,input_width=28)
#X_train, y_train, _, _, _ = util.load_mnist(onehotencoder=False)
#monitor.train(X_train, y_train, classToMonitor, epochs, batch_size, sample_interval, monitors_folder_checkpoint, monitor_name)


'''
#monitoring ensemble of CNNs in the MNIST using outside of box
monitors_ensemble_folder = monitors_folder+"outOfBox_ensembleDNN"+sep
monitor_ensemble_prefix = "monitor_Box_DNN_MNIST"
model_ensemble_prefix = 'DNN_ensemble_MNIST_'
num_cnn = 3
DNN_ensemble_outOfBox_MNIST_monitor.run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, validation_size, K)
'''