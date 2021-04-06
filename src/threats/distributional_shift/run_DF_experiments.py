import os
import logging
from src import model_config
from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.threats.novelty_detection import config
from src.threats.novelty_detection.evaluators import ood_monitor_evaluator
from src.utils import metrics
from src.utils import util
from pathos.multiprocessing import ProcessingPool as Pool
from src import neptune_config as nptne
from src.Classes.dataset import Dataset
from src.threats.novelty_detection.utils import load_monitors
from sklearn import manifold
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


def start(exp_params, save_experiments, parallel_execution, verbose, repetitions, percentage_of_data, log_lvl=logging.FATAL):
	# disabling tensorflow logs
	set_tf_loglevel(log_lvl)
	# re-enabling tensorflow logs
	#set_tf_loglevel(logging.INFO)
	sub_field = exp_params['sub_field']
	if save_experiments:
		nptne.neptune_init(sub_field) # saving experiments in the cloud (optional)

	arr_monitors =  np.array([])
	arr_readouts = []

	model_name = exp_params['model_names']

	## loading experiments
	for modification in exp_params['data']:
		
		# loading dataset
		dataset = Dataset(exp_params['root_dir'])
		dataset.original_dataset_name = exp_params['id_dataset_name']
		dataset.modification = modification
		dataset.dataset_ID_name = exp_params['id_dataset_name']
		dataset.dataset_OOD_name = exp_params['ood_dataset_name']
		dataset_path = os.path.join(dataset.root_dir, sub_field, dataset.original_dataset_name, dataset.modification)
		
		(_, _), (x_test, y_test) = dataset.load_dataset(dataset_path, 'test')
		X, y = x_test, y_test
		
		# for one that wants speeding up tests using part of data
		X_limit = int(len(X)*percentage_of_data)
		y_limit = int(len(y)*percentage_of_data)
		dataset.X, dataset.y = X[ :X_limit], y[ :y_limit]

		# loading model
		backend = exp_params['backend']
		models_folder = os.path.join("src", "bin", "models", backend)
		if backend == 'tensorflow':
			from tensorflow import keras
			model = ModelBuilder(model_name, models_folder)
			model = keras.models.load_model(model.models_folder+'_'+dataset.original_dataset_name+'.h5')

		elif backend == 'keras':
			from keras.models import load_model
			model = ModelBuilder(model_name, models_folder)
			model = load_model(model.models_folder+'_'+dataset.original_dataset_name+'.h5')
		
		elif backend == 'pytorch':
			import torch
			from src.Classes.ml_architectures.pytorch import pytorch_classifiers
			model = ModelBuilder(model_name, models_folder)

			if model_name == 'leNet':
				net = pytorch_classifiers.DNN(exp_params['num_classes_to_monitor_ID'], np.shape(X)[2], 1)

			net.load_state_dict(torch.load(model.models_folder+'_'+dataset.original_dataset_name+'.pth'))
			net.eval()
			model = net

		for technique in exp_params['technique_names']:
			
			PARAMS = config.get_technique_params(technique)	
			
			experiment = Experiment(model_name+'_'+technique)
			#experiment.experiment_type = experiment_type_arg #'OOD' or 'ID'
			experiment.sub_field = sub_field
			experiment.model = model
			experiment.classes_to_monitor_ID = exp_params['num_classes_to_monitor_ID']
			#experiment.classes_to_monitor_OOD = exp_params['num_classes_to_monitor_OOD']
			experiment.dataset = dataset
			experiment.evaluator = ood_monitor_evaluator

			if backend == 'pytorch':
				from src.threats.novelty_detection.testers import ood_tester_pytorch
				experiment.tester = ood_tester_pytorch
			else:
				from src.threats.novelty_detection.testers import ood_tester
				experiment.tester = ood_tester
			
			experiment.verbose = verbose

			monitors = None

			if 'knn' == technique:
				monitors = load_monitors.load_cluster_based_monitors(dataset.original_dataset_name, technique, PARAMS)

			elif 'ocsvm' == technique:
				monitors = load_monitors.load_svm_based_monitors(dataset.original_dataset_name, technique, PARAMS)

			elif 'random_forest' == technique:
				monitors = load_monitors.load_tree_based_monitors(dataset.original_dataset_name, technique, PARAMS)

			elif 'sgd' == technique:
				monitors = load_monitors.load_linear_based_monitors(dataset.original_dataset_name, technique, PARAMS)

			elif technique == 'odin':
				monitors = load_monitors.load_odin_monitor(dataset.original_dataset_name, PARAMS)

			elif technique == 'baseline':
				from src.threats.novelty_detection.evaluators import dnn_baseline_evaluator
				from src.threats.novelty_detection.testers import dnn_baseline_tester
				experiment.evaluator = dnn_baseline_evaluator
				experiment.tester = dnn_baseline_tester

			elif 'oob' in technique:
				monitors = load_monitors.load_box_based_monitors(dataset.original_dataset_name, technique, experiment.classes_to_monitor_ID, PARAMS)

				## diferent evaluator and tester, if ensemble or standalone model
				if 'ensemble' in model_name:
					from src.threats.novelty_detection.testers import en_ood_tester
					experiment.evaluator = en_dnn_oob_evaluator
					experiment.tester = en_ood_tester

			experiment.monitors = monitors
			experiment.PARAMS = PARAMS
			experiment.backend = exp_params['backend']
			
			experiment.evaluator.evaluate(repetitions, experiment, parallel_execution, save_experiments)