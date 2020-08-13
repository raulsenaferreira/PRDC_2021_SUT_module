import os
from src import model_config
from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.novelty_detection.evaluators import dnn_oob_evaluator
from src.novelty_detection.testers import dnn_oob_tester
from src.novelty_detection.testers import en_dnn_oob_tester
from src.novelty_detection.methods import abstraction_box
from src.novelty_detection.methods import act_func_based_monitor
from src.utils import util
from src.utils import metrics
from pathos.multiprocessing import ProcessingPool as Pool
from src.novelty_detection import config as config_ND
from src.Classes.dataset import Dataset
from keras.models import load_model
from src.novelty_detection import load_monitors
from sklearn import manifold
import pickle
import numpy as np


sep = util.get_separator()

# ML is incorrect but monitor does not trust on it = TP
# ML is correct but monitor does not trust on it = FP
# ML is incorrect and monitor trust on it = FN
# ML is correct and monitor trust on it = TN

'''
1 = outside-of-box paper; 2 = outside-of-box using isomap instead of 2D projection; 
3 = outside-of-box with ensemble of DNN; 4 = same of 3 but using isomap strategy;
5 = same of 2 but using DBSCAN instead of KNN; 6 = same of 2 but clustering without dimension reduction;
7 = same of 5 but clustering without dimension reduction; 
8 = using the derivative of activation functions instead of raw values)
'''


def save_results(experiment, arr_readouts, plot=False):
	print("saving experiments", experiment.name)
	filenames = config_ND.load_file_names()
	csvs_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'csv'+sep+experiment.experiment_type+sep+experiment.name+sep
	img_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'img'+sep+experiment.experiment_type+sep+experiment.name+sep

	metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(experiment.name, arr_readouts, img_folder_path+'all_images.pdf')


if __name__ == "__main__":
	# variables regarding Novelty-Detection runtime-monitoring experiments
	experiment_type = 'novelty_detection'
	dataset_names = ['MNIST']#, 'GTSRB']
	validation_size = 0.3
	model_names = config_ND.load_vars(experiment_type, 'model_names')
	technique_names = config_ND.load_vars(experiment_type, 'technique_names')
	num_classes_to_monitor = [10, 43]
	arr_n_components = [2, 3, 5, 10]
	arr_n_clusters_oob = [2, 3, 4, 5]

	# other settings
	parallel_execution = True
	repetitions = 1
	percentage_of_data = 1 #e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data

	## loading experiments
	for model_name, dataset_name, classes_to_monitor in zip(model_names, dataset_names, num_classes_to_monitor):
		arr_monitors =  np.array([])
		pool = Pool()
		processes_pool = []
		# loading dataset
		dataset = Dataset(dataset_name)
		X, y = dataset.load_dataset(mode='test')
		
		# for one that wants speeding up tests using part of data
		X_limit = int(len(X)*percentage_of_data)
		y_limit = int(len(y)*percentage_of_data)
		dataset.X, dataset.y = X[: X_limit], y[: y_limit]

		# loading model
		model = ModelBuilder()
		model = load_model(model.models_folder+model_name+'_'+dataset_name+'.h5')

		#for class_to_monitor in range(classes_to_monitor):
		# loading monitors for Novelty Detection
		for technique in technique_names:
			monitors = load_monitors.load_box_based_monitors(dataset_name, technique, classes_to_monitor,
			 arr_n_clusters_oob, arr_n_components)

			# creating an instance of an experiment
			experiment = Experiment(model_name+'_'+dataset_name+'_monitored_class_')
			experiment.experiment_type = experiment_type
			experiment.dataset = dataset
			experiment.model = model
			experiment.monitors = monitors
			experiment.classes_to_monitor = classes_to_monitor

			## diferent evaluator and tester, if ensemble or standalone model
			if 'ensemble' in model_name:
				experiment.evaluator = en_dnn_oob_evaluator
				experiment.tester = en_dnn_oob_tester
			else:
				experiment.tester = dnn_oob_tester
				experiment.evaluator = dnn_oob_evaluator

			arr_readouts = experiment.evaluator.evaluate(repetitions, experiment, parallel_execution) 
			print('len(arr_readouts)', len(arr_readouts))
			#save_results(experiment, arr_readouts, plot=False)

			'''
			if  parallel_execution:
				processes_pool.append(pool.apipe(experiment.evaluator.evaluate, repetitions, experiment))
			else:
				processes_pool.append(experiment)

			if parallel_execution:
				timeout = 60 * len(experiment.monitors)
				print("\nMax seconds to run each experiment:", timeout)

				for process in processes_pool:
					arr_readouts = process.get(timeout=timeout)	
			else:
				for experiment in processes_pool:
					arr_readouts = experiment.evaluator.evaluate(repetitions, experiment) 
					print('len(arr_readouts)', len(arr_readouts))
					save_results(experiment, arr_readouts, plot=False)
			'''

#print('Class {} with {} monitors on dataset {}'.format(class_to_monitor, len(arr_monitors), dataset_name))
#cd Users\rsenaferre\Desktop\GITHUB\phd_experiments