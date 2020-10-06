import os
import logging
from src import model_config
from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.threats.novelty_detection.evaluators import ood_monitor_evaluator
from src.threats.novelty_detection.evaluators import dnn_baseline_evaluator
from src.threats.novelty_detection.testers import ood_tester
from src.threats.novelty_detection.testers import dnn_baseline_tester
from src.threats.novelty_detection.testers import classifier_based_on_act_func_tester
from src.threats.novelty_detection.testers import en_ood_tester
from src.utils import metrics
from pathos.multiprocessing import ProcessingPool as Pool
from src.threats.novelty_detection import config
from src.Classes.dataset import Dataset
from keras.models import load_model
from src.threats.novelty_detection import load_monitors
from sklearn import manifold
import pickle
import numpy as np
import argparse

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

'''
def save_results(PARAMS, classes_to_monitor_ID, sub_field, name, technique, arr_readouts, plot=False):
	print("saving experiments", name)
	filenames = config_ND.load_file_names()
	csvs_folder_path = os.path.join('src', 'tests', 'results', 'csv', sub_field, name, '_'+technique)
	img_folder_path = os.path.join('src', 'tests', 'results', 'img', sub_field, name, '_'+technique)

	#metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	metrics.save_results_in_neptune(PARAMS, arr_readouts, classes_to_monitor_ID)
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(name, arr_readouts, img_folder_path+'all_images.pdf')
'''

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]	


def start(sub_field, save_experiments, parallel_execution, verbose, repetitions, percentage_of_data, log_lvl=logging.FATAL):
	exp_params = config.get_experiment_params(1)
	# disabling tensorflow logs
	set_tf_loglevel(log_lvl)
	# re-enabling tensorflow logs
	#set_tf_loglevel(logging.INFO)

	if save_experiments:
		config.neptune_init() # saving experiments in the cloud (optional)

	## loading experiments
	for model_name, dataset_name, classes_to_monitor_ID in zip(exp_params['model_names'],\
	 exp_params['dataset_names'], exp_params['arr_classes_to_monitor_ID']):
		arr_monitors =  np.array([])
		arr_readouts = []

		# loading ID dataset
		dataset = Dataset(dataset_name)
		X, y = dataset.load_dataset(mode='test')

		#loading OOD dataset
		OOD_dataset = Dataset(ood_dataset_name)
		
		ood_X, ood_y = OOD_dataset.load_dataset(mode='test_entire_data')
		ood_y += classes_to_monitor_ID #avoiding same class numbers for the two datasets

		#concatenate and shuffling ID and OOD datasets
		X = np.vstack([X, ood_X])
		y = np.hstack([y, ood_y])
		X, y = unison_shuffled_copies(X, y)

		print("Final dataset shape", X.shape, y.shape)
		dataset.dataset_ID_name = dataset_name
		dataset.dataset_OOD_name = ood_dataset_name
		
		# for one that wants speeding up tests using part of data
		X_limit = int(len(X)*percentage_of_data)
		y_limit = int(len(y)*percentage_of_data)
		dataset.X, dataset.y = X[: X_limit], y[: y_limit]

		# loading model
		model = ModelBuilder(model_name)
		model = load_model(model.models_folder+'_'+dataset_name+'.h5')

		for technique in exp_params['technique_names']:
			
			PARAMS = config.get_technique_params(technique)	
			
			experiment = Experiment(model_name+'_'+technique)
			#experiment.experiment_type = experiment_type_arg #'OOD' or 'ID'
			experiment.sub_field = sub_field
			experiment.model = model
			experiment.classes_to_monitor_ID = classes_to_monitor_ID
			experiment.classes_to_monitor_OOD = ood_num_classes_to_monitor
			experiment.dataset = dataset
			experiment.evaluator = ood_monitor_evaluator
			experiment.tester = ood_tester
			#experiment.tester = classifier_based_on_act_func_tester
			experiment.verbose = verbose

			monitors = None

			if 'knn' == technique:
				monitors = load_monitors.load_cluster_based_monitors(dataset_name, technique, PARAMS)

			elif 'ocsvm' == technique:
				monitors = load_monitors.load_svm_based_monitors(dataset_name, technique, PARAMS)

			elif 'random_forest' == technique:
				monitors = load_monitors.load_tree_based_monitors(dataset_name, technique, PARAMS)

			elif 'sgd' == technique:
				monitors = load_monitors.load_linear_based_monitors(dataset_name, technique, PARAMS)

			elif technique == 'baseline':
				experiment.evaluator = dnn_baseline_evaluator
				experiment.tester = dnn_baseline_tester

			elif 'oob' in technique:
				monitors = load_monitors.load_box_based_monitors(dataset_name, technique, classes_to_monitor_ID, PARAMS)

				## diferent evaluator and tester, if ensemble or standalone model
				if 'ensemble' in model_name:
					experiment.evaluator = en_dnn_oob_evaluator
					experiment.tester = en_ood_tester

			experiment.monitors = monitors
			experiment.PARAMS = PARAMS
			
			experiment.evaluator.evaluate(repetitions, experiment, parallel_execution, save_experiments)