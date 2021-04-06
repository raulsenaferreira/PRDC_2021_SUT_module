import os 
from sklearn import manifold
from src.Classes.monitor import Monitor
from src.threats.novelty_detection.methods import abstraction_box
from sklearn.decomposition import PCA
import pickle
import numpy as np
from pathlib import Path
from src.threats.novelty_detection.methods.sota.ODIN import odin_monitor 
from src.threats.novelty_detection.methods.sota.ALOCC_Keras import models as ALOOC


dir_path = os.path.dirname(Path(__file__).parent)
root_path = os.path.join(dir_path, 'bin', 'monitors')
#dir_path = os.path.join('\\src', 'threats', 'novelty_detection')

monitoring_characteristics = 'dnn_internals'


def create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics, PARAMS):
	monitor = Monitor(monitor_name)

	# for class_to_monitor in range(classes_to_monitor):
	monitor_folder = os.path.join(root_path ,  monitoring_characteristics ,  dataset_name, technique)
	# monitor_folder += technique +sep+ 'class_'+str(class_to_monitor) +sep
	#monitor_folder += technique
	monitor.monitors_folder = monitor_folder
	
	if PARAMS['use_alternative_monitor']:
		monitor.filename = 'monitor_'+monitor_name+'.p_2'
	else:
		monitor.filename = 'monitor_'+monitor_name+'.p'

	#if PARAMS['use_scaler']:
	#	monitor.filename = '(scaled_input_version)'+monitor.filename

	monitor.use_alternative_monitor = PARAMS['use_alternative_monitor']
	monitor.use_scaler = PARAMS['use_scaler']

	return monitor


def load_cluster_based_monitors(dataset_name, technique, PARAMS):

	monitors = []

	if 'knn' == technique:
		arr_n_clusters = PARAMS['arr_n_clusters']

		for n_clusters in arr_n_clusters:
			monitor_name = technique+'_{}_clusters'.format(n_clusters)
			monitor = create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics)
			monitor.n_clusters = n_clusters
			monitor.use_scaler = PARAMS['use_scaler']

			monitors.append(monitor)

	return np.array(monitors)


def load_tree_based_monitors(dataset_name, technique, PARAMS):
	
	if 'random_forest' == technique:
		if PARAMS['grid_search']:
			monitor_name = technique+'_optimized'
		else:
			monitor_name = technique+'_not_optimized'
		monitor = create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics, PARAMS)
		
	return np.array([monitor])


def load_linear_based_monitors(dataset_name, technique, PARAMS):
	
	if 'sgd' == technique:
		if PARAMS['grid_search']:
			monitor_name = technique+'_optimized'
		else:
			monitor_name = technique+'_not_optimized'
		monitor = create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics, PARAMS)
		monitor.OOD_approach = PARAMS['OOD_approach']

	return np.array([monitor])


def load_odin_monitor(dataset_name, PARAMS):
	monitor_name = 'odin'
	monitor = Monitor(monitor_name)

	monitor_folder = os.path.join(root_path,  monitoring_characteristics,  dataset_name, monitor_name)
	
	monitor.monitors_folder = monitor_folder
	
	monitor.filename = 'monitor_odin.p'
	# line not necessary in the final version
	monitor.method = odin_monitor #odin_keras

	monitor.use_alternative_monitor = PARAMS['use_alternative_monitor']
	monitor.use_scaler = PARAMS['use_scaler']
	monitor.OOD_approach = PARAMS['OOD_approach']
	monitor.noiseMagnitude = PARAMS['noiseMagnitude']
	monitor.temper = PARAMS['temper']
	monitor.threshold = PARAMS['threshold']
	
	return np.array([monitor])


def load_alocc_monitor(dataset_name, PARAMS):

	monitor_name = 'alooc'
	monitor = Monitor(monitor_name)

	monitor.OOD_approach = PARAMS['OOD_approach']
	monitor.use_scaler = PARAMS['use_scaler']
	monitor.optimizer = PARAMS['optimizer']
	monitor.model_number = PARAMS['model_number']
	monitor.threshold = PARAMS['threshold']

	alooc_monitor = ALOOC.ALOCC_Model(input_height=PARAMS['input_height'], input_width=PARAMS['input_width'],
		     output_height=PARAMS['output_height'], output_width=PARAMS['output_width'], c_dim = 3)

	monitor_folder = os.path.join(root_path,  monitoring_characteristics,  dataset_name, monitor_name, monitor.optimizer)
	
	monitor.monitors_folder = monitor_folder
	
	monitor.filename = 'monitor_alooc.p'
	# line not necessary in the final version
	monitor.method = alooc_monitor 
	
	return np.array([monitor])


def load_svm_based_monitors(dataset_name, technique, PARAMS):
	
	if 'ocsvm' == technique:
		if PARAMS['grid_search']:
			monitor_name = technique+'_optimized'
		else:
			monitor_name = technique+'_not_optimized'
		monitor = create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics, PARAMS)
		monitor.OOD_approach = 'outlier' # only possible using the outlier approach

	return np.array([monitor])


def load_box_based_monitors(dataset_name, technique, classes_to_monitor, params):

	monitors = []
	arr_n_clusters_oob = params['arr_n_clusters']
	tau = params['tau']
	use_alternative_monitor = params['use_alternative_monitor']
	use_scaler = params['use_scaler']
	OOD_approach = params['OOD_approach']
	
	for n_clusters_oob in arr_n_clusters_oob:
		#for tau in arr_tau_oob:
		monitor = None
		boxes = {}

		if 'oob' == technique:
			monitor_name = technique+'_{}_clusters'.format(n_clusters_oob)
			monitor = Monitor(monitor_name)
			monitor.n_clusters = n_clusters_oob

			#for class_to_monitor in range(classes_to_monitor):
			monitor_folder = os.path.join(root_path, monitoring_characteristics ,  dataset_name)
			#	monitor_folder += technique +sep+ 'class_'+str(class_to_monitor) +sep
			monitor_folder = os.path.join(monitor_folder, technique, 'class_')
			monitor.monitors_folder = monitor_folder

			if use_alternative_monitor:
				monitor.filename = 'monitor_'+monitor_name+'.p_2' #built with true labels instead of right predictions
			else:
				monitor.filename = 'monitor_'+monitor_name+'.p'
			
				#monitor_path = monitor.monitors_folder+monitor.filename
				# loading abstraction boxes
				#boxes[class_to_monitor] = pickle.load(open(monitor_path, "rb"))

			if 'ensemble' in technique:
				monitor.method = abstraction_box.find_point_box_ensemble
			else:
				#monitor.method = abstraction_box.find_point
				monitor.method = abstraction_box.check_outside_of_box

			monitor.dim_reduc_method = None
			monitor.tau = tau
			monitor.use_scaler = use_scaler
			monitor.OOD_approach = OOD_approach
			monitors.append(monitor)
			
		elif 'oob_isomap' == technique or 'oob_pca' == technique or 'oob_pca_isomap' == technique:

			arr_n_components = params['arr_n_components']

			for n_components in arr_n_components:
				boxes = {}
				dim_reduc_method = []
				
				monitor_name = technique+'_{}_components_{}_clusters'.format(n_components, n_clusters_oob)
				reduc_name = technique+'_{}_components'.format(n_components)

				monitor = Monitor(monitor_name)
				monitor.n_clusters = n_clusters_oob

				#for class_to_monitor in range(classes_to_monitor):
				monitor_folder = os.path.join(root_path ,  monitoring_characteristics ,  dataset_name)
				#	monitor_folder += technique +sep+ 'class_'+str(class_to_monitor) +sep
				monitor_folder = os.path.join(monitor_folder, technique ,  'class_')
				monitor.monitors_folder = monitor_folder

				if use_alternative_monitor:
					monitor.filename = 'monitor_'+monitor_name+'.p_2' #built with true labels instead of right predictions
				else:
					monitor.filename = 'monitor_'+monitor_name+'.p'

				#	dim_reduc_method[class_to_monitor] = pickle.load(open(monitor_folder+'trained_'+reduc_name+'.p', "rb"))
				
					# loading abstraction boxes
					#monitor_path = monitor.monitors_folder+monitor.filename
					#boxes[class_to_monitor] = pickle.load(open(monitor_path, "rb"))
				
				if 'ensemble' in technique:
					monitor.method = abstraction_box.find_point_box_ensemble
				else:
					#monitor.method = abstraction_box.find_point
					monitor.method = abstraction_box.check_outside_of_box

				#monitor.dim_reduc_method = dim_reduc_method
				monitor.dim_reduc_method = reduc_name

				if 'oob_pca_isomap' == technique:
					dim_reduc_method.append('PCA_'+reduc_name+'.p')
					dim_reduc_method.append('Isomap_'+reduc_name+'.p')
					monitor.dim_reduc_method = dim_reduc_method

				monitor.tau = tau
				monitor.use_scaler = use_scaler
				monitor.OOD_approach = OOD_approach
				monitors.append(monitor)

		elif '' == technique:
			pass

	return np.array(monitors)