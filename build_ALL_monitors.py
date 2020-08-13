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


def build_all_monitors(parallel_execution, arr_n_components, arr_n_clusters_oob,
 technique_names, classes_to_monitor, model_file, X, y):
	arr_monitors = []

	# Generate monitors for each class for a specific dataset
	for class_to_monitor in range(classes_to_monitor):
		#Building monitors for Novelty Detection
		monitors = build_monitors.prepare_box_based_monitors(dataset_name, technique_names, class_to_monitor,
		 arr_n_clusters_oob, arr_n_components)
		arr_monitors.extend(monitors)
		#arr_monitors = np.append(arr_monitors, monitors)
	
	#Parallelizing the experiments (optional): one experiment per Core
	if parallel_execution:
		timeout = 30 * len(arr_monitors)
		pool = Pool()
		processes_pool = []

		for monitor in arr_monitors:
			processes_pool.append(pool.apipe(monitor.trainer.run, monitor, model_file, X, y)) 
		
		for process in processes_pool:
			trained_monitor = process.get(timeout=timeout)
	else:
		for monitor in arr_monitors:
			trained_monitor = monitor.trainer.run(monitor, model_file, X, y)
			


if __name__ == "__main__":
	#general settings
	sep = util.get_separator()
	parallel_execution = True
	parallel_execution_on_CPU = True
	parallel_execution_on_GPU = True

	arr_n_components = [2, 3, 5, 10]
	arr_n_clusters_oob = [2, 3, 4, 5]

	experiment_type = 'novelty_detection'
	dataset_names = ['MNIST','GTSRB']# 
	validation_size = 0.3
	model_names = config_ND.load_vars(experiment_type, 'model_names')
	technique_names = config_ND.load_vars(experiment_type, 'technique_names')
	num_classes_to_monitor = [10, 43]# 
	perc_of_data = 1 #e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data
	
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

		build_all_monitors(parallel_execution, arr_n_components, 
			arr_n_clusters_oob, technique_names, classes_to_monitor, model, X, y)
		
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