import os
import logging
from src import model_config
from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.novelty_detection.evaluators import dnn_oob_evaluator
from src.novelty_detection.testers import dnn_oob_tester
from src.novelty_detection.testers import en_dnn_oob_tester
from src.novelty_detection.evaluators import baseline_evaluator
from src.novelty_detection.testers import baseline_tester
from src.novelty_detection.evaluators import cluster_based_act_func_evaluator
from src.novelty_detection.testers import cluster_based_act_func_tester
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
import neptune


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


def save_results(PARAMS, classes_to_monitor, sub_field, name, technique, arr_readouts, plot=False):
	print("saving experiments", name)
	filenames = config_ND.load_file_names()
	csvs_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'csv'+sep+sub_field+sep+name+sep+'_'+technique+sep
	img_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'img'+sep+sub_field+sep+name+sep+'_'+technique+sep

	#metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	metrics.save_results_in_neptune(PARAMS, arr_readouts, classes_to_monitor)
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(name, arr_readouts, img_folder_path+'all_images.pdf')


if __name__ == "__main__":
	# variables regarding Novelty-Detection runtime-monitoring experiments
	sub_field = 'novelty_detection'
	dataset_names = ['GTSRB'] #'MNIST', 'GTSRB'
	validation_size = 0.3
	model_names = ['leNet'] #, 'leNet'
	num_classes_to_monitor = [43] #10, 
	
	PARAMS = {'arr_n_components' : [2], #2, 3, 5, 10
	 'arr_n_clusters' : [2, 3, 5, 10], #1, 2, 3, 4, 5
	 #for dbscan
	 'eps' : [0.2, 0.3, 0.5], 'min_samples': [5],  #min_samples 3, 5, 7, 10

	 'technique_names' : ['dbscan']}#'baseline', 'knn', 'oob', 'oob_isomap', 'oob_pca', 'oob_pca_isomap'

	# other settings
	save_experiments = True
	parallel_execution = False
	repetitions = 1
	percentage_of_data = 1 #e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data

	# disabling tensorflow logs
	set_tf_loglevel(logging.FATAL)
	# re-enabling tensorflow logs
	#set_tf_loglevel(logging.INFO)

	if save_experiments:
		#saving experiments in the cloud (optional)
		neptune.init('raulsenaferreira/PhD')

	## loading experiments
	for model_name, dataset_name, classes_to_monitor in zip(model_names, dataset_names, num_classes_to_monitor):
		arr_monitors =  np.array([])
		arr_readouts = []

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
		for technique in PARAMS['technique_names']:
			# creating an instance of an experiment
			experiment = Experiment(model_name+'_'+dataset_name)
			experiment.experiment_type = 'ID'
			experiment.sub_field = sub_field
			experiment.model = model
			experiment.classes_to_monitor = classes_to_monitor
			experiment.dataset = dataset

			monitors = None

			if technique == 'baseline':
				experiment.evaluator = baseline_evaluator
				experiment.tester = baseline_tester

			elif 'oob' in technique:
				monitors = load_monitors.load_box_based_monitors(dataset_name, technique, classes_to_monitor, PARAMS)

				## diferent evaluator and tester, if ensemble or standalone model
				if 'ensemble' in model_name:
					experiment.evaluator = en_dnn_oob_evaluator
					experiment.tester = en_dnn_oob_tester
				else:
					experiment.tester = dnn_oob_tester
					experiment.evaluator = dnn_oob_evaluator

			elif 'knn' == technique or 'dbscan' == technique:
				monitors = load_monitors.load_cluster_based_monitors(dataset_name, technique, PARAMS)
				experiment.evaluator = cluster_based_act_func_evaluator
				experiment.tester = cluster_based_act_func_tester

			experiment.monitors = monitors
			
			experiment.evaluator.evaluate(repetitions, experiment, parallel_execution, save_experiments)
			#print('len(arr_readouts)', len(readouts))
			#arr_readouts.append(readouts)
		
			#save_results(PARAMS, classes_to_monitor, experiment.sub_field, experiment.name, technique, arr_readouts, plot=False)

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