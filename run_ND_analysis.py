import os
import logging
from src import model_config
from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.novelty_detection.evaluators import ood_monitor_evaluator
from src.novelty_detection.evaluators import dnn_baseline_evaluator
from src.novelty_detection.testers import dnn_oob_tester
from src.novelty_detection.testers import dnn_baseline_tester
from src.novelty_detection.testers import classifier_based_on_act_func_tester
from src.novelty_detection.testers import en_dnn_oob_tester
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
import argparse


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


def save_results(PARAMS, classes_to_monitor_ID, sub_field, name, technique, arr_readouts, plot=False):
	print("saving experiments", name)
	filenames = config_ND.load_file_names()
	csvs_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'csv'+sep+sub_field+sep+name+sep+'_'+technique+sep
	img_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'img'+sep+sub_field+sep+name+sep+'_'+technique+sep

	#metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	metrics.save_results_in_neptune(PARAMS, arr_readouts, classes_to_monitor_ID)
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(name, arr_readouts, img_folder_path+'all_images.pdf')


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



def start(experiment_type_arg, save_experiments, parallel_execution, repetitions, percentage_of_data):
	
	# experiments regarding Novelty-Detection runtime-monitoring experiments
	sub_field = 'novelty_detection'

	dataset_names = ['GTSRB'] #'MNIST', 'GTSRB'
	arr_classes_to_monitor_ID = [43] #10, 

	ood_dataset_name = 'BTSC'
	ood_num_classes_to_monitor = 62

	model_names = ['leNet'] #, 'leNet'
	
	PARAMS = {'arr_n_components' : [2], #2, 3, 5, 10
	 'arr_n_clusters' : [2, 3, 5, 10], #1, 2, 3, 4, 5
	 #for hdbscan
	 'min_samples': [5],  #min_samples 5, 10, 15
	 #for knn and sgd classifiers
	 'use_scaler': False,
	 'OOD_approach': 'equality', # 'equality', 'outlier'
	 #for all methods
	 'use_alternative_monitor': False,# True = label -> act func; False = label -> act func if label == predicted
	 'technique_names' : ['ocsvm']}#'baseline', 'knn', 'random_forest', 'sgd', 'hdbscan', 'oob', 'oob_isomap', 'oob_pca', 'oob_pca_isomap'

	# disabling tensorflow logs
	set_tf_loglevel(logging.FATAL)
	# re-enabling tensorflow logs
	#set_tf_loglevel(logging.INFO)

	if save_experiments:
		#saving experiments in the cloud (optional)
		neptune.init('raulsenaferreira/PhD')

	## loading experiments
	for model_name, dataset_name, classes_to_monitor_ID in zip(model_names, dataset_names, arr_classes_to_monitor_ID):
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
			experiment = Experiment(model_name+'_'+dataset_name)
			experiment.experiment_type = experiment_type_arg #'OOD' or 'ID'
			experiment.sub_field = sub_field
			experiment.model = model
			experiment.classes_to_monitor_ID = classes_to_monitor_ID
			experiment.classes_to_monitor_OOD = ood_num_classes_to_monitor
			experiment.dataset = dataset

			monitors = None

			if 'knn' == technique:
				monitors = load_monitors.load_cluster_based_monitors(dataset_name, technique, PARAMS)

			elif 'ocsvm' == technique:
				monitors = load_monitors.load_svm_based_monitors(dataset_name, technique, PARAMS)

			elif 'random_forest' == technique:
				monitors = load_monitors.load_tree_based_monitors(dataset_name, technique, PARAMS)

			elif 'sgd' == technique:
				monitors = load_monitors.load_linear_based_monitors(dataset_name, technique, PARAMS)

			experiment.evaluator = ood_monitor_evaluator
			experiment.tester = classifier_based_on_act_func_tester

			elif technique == 'baseline':
				experiment.evaluator = dnn_baseline_evaluator
				experiment.tester = dnn_baseline_tester

			elif 'oob' in technique:
				monitors = load_monitors.load_box_based_monitors(dataset_name, technique, classes_to_monitor_ID, PARAMS)

				## diferent evaluator and tester, if ensemble or standalone model
				if 'ensemble' in model_name:
					experiment.evaluator = en_dnn_oob_evaluator
					experiment.tester = en_dnn_oob_tester
				else:
					experiment.tester = dnn_oob_tester
					experiment.evaluator = ood_monitor_evaluator

			experiment.monitors = monitors
			
			experiment.evaluator.evaluate(repetitions, experiment, parallel_execution, save_experiments) 

#if __name__ == "__main__":
#	start()