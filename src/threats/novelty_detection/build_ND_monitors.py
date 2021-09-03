import os
import logging
import numpy as np
from src import model_config as model_cfg
from src.Classes.model_builder import ModelBuilder
from pathos.multiprocessing import ProcessingPool as Pool
from src.Classes.dataset import Dataset
from src.threats.novelty_detection.utils import create_monitors
from timeit import default_timer as timer



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


def build_monitors_by_class(root_path, parallel_execution, dataset_name, params, classes_to_monitor, model, X, y, save):
	#arr_monitors = []

	# Generate monitors for each class for a specific dataset
	'''for class_to_monitor in range(classes_to_monitor):
		monitors = create_monitors.prepare_box_based_monitors(root_path, dataset_name, params, class_to_monitor)
		arr_monitors.extend(monitors)
		#arr_monitors = np.append(arr_monitors, monitors)
	'''
	arr_monitors = {}
	technique_names = params['technique_names']
	arr_n_components = params['arr_n_components']
	arr_n_clusters_oob = params['arr_n_clusters']

	for technique_name in technique_names:
		
		monitors_by_class = {}
		for class_to_monitor in range(classes_to_monitor):
			monitor = create_monitors.prepare_box_based_monitors(root_path, dataset_name,\
			 technique_name, arr_n_clusters_oob, arr_n_components, class_to_monitor)
			monitors_by_class.update({class_to_monitor: monitor})

		arr_monitors.update({technique_name: monitors_by_class})

	#Parallelizing the experiments (optional): one experiment per Core
	if parallel_execution:
		cores = 6
		timeout = 30 * len(arr_monitors)
		pool = Pool(cores)
		processes_pool = []

		print("\nParallel execution with {} cores. Max {} seconds to run each experiment:".format(cores, timeout))

		for monitor in arr_monitors:
			processes_pool.append(pool.apipe(monitor.trainer.run, monitor, model, X, y, save, params)) 
		
		for process in processes_pool:
			_ = process.get(timeout=timeout)
	else:
		print("\nSerial execution.")
		#for monitor in arr_monitors:
		#	_ = monitor.trainer.run(monitor, model, X, y, save, params)
		for technique, monitors_by_class in arr_monitors.items():
			train = monitors_by_class[0].trainer
			_ = train.run(monitors_by_class, model, X, y, save, params)
			

def build_monitors_all_classes(root_path, parallel_execution, dataset_name, params, model, X, y, save):
	# Generate monitors for all classes for a specific dataset
	arr_monitors = create_monitors.build_monitors(root_path, dataset_name, params)
	
	#Parallelizing the experiments (optional): one experiment per Core
	if parallel_execution:
		cores = 6
		timeout = 30 * len(arr_monitors)
		pool = Pool(cores)
		processes_pool = []

		print("\nParallel execution with {} cores. Max {} seconds to run each experiment:".format(cores, timeout))

		for monitor in arr_monitors:
			processes_pool.append(pool.apipe(monitor.trainer.run, monitor, model, X, y, save, params)) 
		
		for process in processes_pool:
			process.get(timeout=timeout)
	else:
		print("\nSerial execution.")
		for monitor in arr_monitors:
			monitor.trainer.run(monitor, model, X, y, save, params)


def start(sub_field, DATA_PARAMS, MONITOR_PARAMS, save, parallel_execution, root_path, perc_of_data):
	# disabling tensorflow logs
	set_tf_loglevel(logging.FATAL)
	# re-enabling tensorflow logs
	#set_tf_loglevel(logging.INFO)

	dataset_folder = DATA_PARAMS['dataset_folder']
	dataset_names = DATA_PARAMS['dataset_names']
	num_classes_to_monitor = DATA_PARAMS['num_classes_to_monitor']
	validation_size = DATA_PARAMS['validation_size']

	model_names = MONITOR_PARAMS['model_names']
	is_build_monitors_by_class = MONITOR_PARAMS['is_build_monitors_by_class']
	
	for model_name, dataset_name, classes_to_monitor in zip(model_names, dataset_names, num_classes_to_monitor):
		
		# loading dataset
		dataset = Dataset(dataset_name)
		dataset.validation_size = validation_size

		MONITOR_PARAMS.update({'dataset_name': dataset_name})
		MONITOR_PARAMS.update({'model_name': model_name})
		
		#building monitor with training (test data is excluded, of course)
		path = os.path.join(dataset_folder, sub_field, dataset_name)
		(x_train, y_train), (_, _) = dataset.load_dataset(path)
		X, y = x_train[:int(len(x_train)*perc_of_data)], y_train[:int(len(y_train)*perc_of_data)]

		#path to load the model
		models_folder = os.path.join("src", "bin", "models", MONITOR_PARAMS['backend'])
		model_file = os.path.join(models_folder, model_name+'_'+dataset_name)
		
		# loading model
		if MONITOR_PARAMS['backend']=='tensorflow':
			from tensorflow import keras
			model = keras.models.load_model(model_file+'.h5')
		
		elif MONITOR_PARAMS['backend']=='keras':
			from keras.models import load_model
			model = load_model(model_file+'.h5')

		elif MONITOR_PARAMS['backend']=='pytorch':
			import torch
			from src.Classes.ml_architectures.pytorch import pytorch_classifiers

			if model_name == 'leNet':
				net = pytorch_classifiers.DNN(classes_to_monitor, np.shape(X)[2], 1)
			else:
				net = None

			net.load_state_dict(torch.load(model_file+'.pth'))
			net.eval()
			model = net

				
		start = timer()

		if is_build_monitors_by_class:
			build_monitors_by_class(root_path, parallel_execution, dataset_name, MONITOR_PARAMS, classes_to_monitor, model, X, y, save)
		else:
			build_monitors_all_classes(root_path, parallel_execution, dataset_name, MONITOR_PARAMS, model, X, y, save)
		
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