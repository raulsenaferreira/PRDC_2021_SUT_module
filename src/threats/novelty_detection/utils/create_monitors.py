import os
import numpy as np
from src.Classes.monitor import Monitor
from src.threats.novelty_detection.methods import abstraction_box
from src.threats.novelty_detection.methods import act_func_based_monitor
from src.threats.novelty_detection.methods import clustered_act_function_monitor
from src.threats.novelty_detection.methods import tree_based_act_function_monitor
from src.threats.novelty_detection.methods import linear_based_act_function_monitor
from src.threats.novelty_detection.methods import ocsvm_based_act_function_monitor
from src.threats.novelty_detection.methods.sota.ODIN import threshold_finder 
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap



def build_abstraction_based_monitor(class_to_monitor, monitor_name, n_clusters_oob, monitor_folder):
	monitor = Monitor(monitor_name)
	monitor.class_to_monitor = class_to_monitor
	monitor.trainer = act_func_based_monitor
	monitor.method = abstraction_box.make_abstraction
	monitor.filename = 'monitor_'+monitor_name+'.p'
	monitor.n_clusters = n_clusters_oob
	monitor.monitors_folder = monitor_folder

	return monitor


def build_gradient_based_monitor(class_to_monitor, monitor_name, n_clusters_oob, monitor_folder):
	monitor = Monitor(monitor_name)
	monitor.class_to_monitor = class_to_monitor
	monitor.trainer = act_func_gradient_based_monitor
	monitor.method = abstraction_box.make_abstraction
	monitor.filename = 'monitor_'+monitor_name+'.p'
	monitor.n_clusters = n_clusters_oob
	monitor.monitors_folder = monitor_folder

	return monitor


def build_classifier_based_monitor(technique, trainer, monitor_name, monitor_folder):
	monitor = Monitor(monitor_name)
	monitor.trainer = trainer
	monitor.method = technique
	monitor.filename = 'monitor_'+monitor_name+'.p'
	monitor.monitors_folder = monitor_folder

	return monitor


def build_ODIN_based_monitor(technique, trainer, PARAMS, monitor_folder):
	monitor = Monitor(technique)
	monitor.trainer = trainer
	monitor.method = technique
	monitor.filename = 'monitor_'+technique+'.p'
	monitor.monitors_folder = monitor_folder
	monitor.magnitude = PARAMS['magnitude']
	monitor.temperature = PARAMS['temperature']

	return monitor


def prepare_monitors(technique, trainer, monitor_folder, dataset_name, PARAMS):
	monitors = []
	
	use_grid_search = PARAMS['use_grid_search']
	monitor_name = technique+'_not_optimized'

	if use_grid_search:
		monitor_name = technique+'_optimized'

	if technique == 'odin':
		monitor = build_ODIN_based_monitor(technique, trainer, PARAMS, monitor_folder)
	else:
		monitor = build_classifier_based_monitor(technique, trainer, monitor_name, monitor_folder)

	monitor.use_grid_search = use_grid_search
	monitor.use_alternative_monitor = PARAMS['use_alternative_monitor']
	monitor.use_scaler = PARAMS['use_scaler']

	monitors.append(monitor)

	return monitors


def prepare_cluster_based_monitors(technique, monitor_folder, dataset_name, PARAMS):
	monitors = []
	if technique  == 'knn':
		arr_n_clusters = PARAMS['arr_n_clusters']
		for n_clusters in arr_n_clusters:
			monitor_name = technique+'_{}_clusters'.format(n_clusters)			
			monitor = build_classifier_based_monitor(technique, clustered_act_function_monitor, monitor_name, monitor_folder)
			monitor.n_clusters = n_clusters

			monitor.use_alternative_monitor = PARAMS['use_alternative_monitor']
			monitor.use_scaler = PARAMS['use_scaler']

			monitors.append(monitor)

	return monitors


def prepare_box_based_monitors(root_path, dataset_name, technique, n_clusters_oob, n_components, class_to_monitor):
	monitoring_characteristics = 'dnn_internals'
	#monitors = []
	c = '{}'.format(class_to_monitor)

	monitor_folder = os.path.join(root_path, monitoring_characteristics, dataset_name, technique, 'class_'+ c)
	print('creating monitor {} for class {}'.format(technique, c))
		
	if 'oob' == technique:
		monitor_name = technique+'_{}_clusters'.format(n_clusters_oob)
		
		monitor = build_abstraction_based_monitor(class_to_monitor, monitor_name, n_clusters_oob, monitor_folder)

		#monitors.append(monitor)
		return monitor

	elif 'oob_gradient' == technique:
		monitor_name = technique+'_{}_clusters'.format(n_clusters_oob)
		
		monitor = build_gradient_based_monitor(class_to_monitor, monitor_name, n_clusters_oob, monitor_folder)

		#monitors.append(monitor)
		return monitor
	
	elif 'oob_isomap' == technique or 'oob_pca' == technique or 'oob_pca_isomap' == technique:
		
		dim_reduc_method = []
		dim_reduc_filename_prefix = []

		monitor_name = technique+'_{}_components_{}_clusters'.format(n_components, n_clusters_oob)
		reduc_name = technique+'_{}_components'.format(n_components)

		if 'oob_isomap' == technique:
			dim_reduc_method = Isomap(n_components = n_components)
			dim_reduc_filename_prefix = 'trained_'+reduc_name+'.p'
		elif 'oob_pca' == technique:
			dim_reduc_method = PCA(n_components = n_components)
			dim_reduc_filename_prefix = 'trained_'+reduc_name+'.p'
		elif 'oob_pca_isomap' == technique:
			dim_reduc_method.append(PCA(n_components = 20)) #good range value for image datasets: (20-40)
			dim_reduc_filename_prefix.append('trained_PCA_'+reduc_name+'.p')
			dim_reduc_method.append(Isomap(n_components=n_components))
			dim_reduc_filename_prefix.append('trained_Isomap_'+reduc_name+'.p')

		monitor = build_abstraction_based_monitor(class_to_monitor, monitor_name, n_clusters_oob, monitor_folder)
		monitor.n_components = n_components
		monitor.dim_reduc_filename_prefix = dim_reduc_filename_prefix
		monitor.technique = technique
		monitor.dim_reduc_method = dim_reduc_method
		
		#monitors.append(monitor)
		return monitor

	#return monitors


def build_monitors(root_path, dataset_name, params):
	monitoring_characteristics = 'dnn_internals'
	arr_monitors = []
	technique_names = params['technique_names']
	
	for technique in technique_names:
		trainer = None
		monitor_folder = os.path.join(root_path, monitoring_characteristics, dataset_name, technique)
		#monitor_folder += technique +sep

		if technique  == 'knn':
			trainer = clustered_act_function_monitor
		elif technique == 'random_forest':
			trainer = tree_based_act_function_monitor
		elif technique == 'sgd':
			trainer = linear_based_act_function_monitor
		elif technique == 'ocsvm':
			trainer = ocsvm_based_act_function_monitor
		elif technique == 'odin':
			trainer = threshold_finder

		monitors = prepare_monitors(technique, trainer, monitor_folder, dataset_name, params)

		arr_monitors.extend(monitors)
	return arr_monitors