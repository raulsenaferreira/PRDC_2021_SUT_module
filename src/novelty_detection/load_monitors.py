import os 
from src.utils import util
from sklearn import manifold
from src.Classes.monitor import Monitor
from src.novelty_detection.methods import abstraction_box
from sklearn.decomposition import PCA
import pickle
import numpy as np


sep = util.get_separator()
dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = dir_path+sep+'bin'+sep+'monitors'


def create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics):
	monitor = Monitor(monitor_name)

	# for class_to_monitor in range(classes_to_monitor):
	monitor_folder = root_path +sep+ monitoring_characteristics +sep+ dataset_name +sep
	# monitor_folder += technique +sep+ 'class_'+str(class_to_monitor) +sep
	monitor_folder += technique +sep
	monitor.monitors_folder = monitor_folder
	monitor.filename = 'monitor_'+monitor_name+'.p'

	return monitor


def load_cluster_based_monitors(dataset_name, technique, PARAMS):
	monitoring_characteristics = 'dnn_internals'
	monitors = []

	if 'knn' == technique:
		arr_n_clusters = PARAMS['arr_n_clusters']

		for n_clusters in arr_n_clusters:
			monitor_name = technique+'_{}_clusters'.format(n_clusters)
			monitor = create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics)
			monitor.n_clusters = n_clusters
			monitors.append(monitor)

	elif 'hdbscan' == technique:
		arr_min_samples = PARAMS['min_samples']

		for min_samples in arr_min_samples:
			monitor_name = technique+'_{}_min_samples'.format(min_samples)
			monitor = create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics)
			monitor.min_samples = min_samples
			monitors.append(monitor)

	return np.array(monitors)


def load_tree_based_monitors(dataset_name, technique, PARAMS):
	monitoring_characteristics = 'dnn_internals'
	if 'random_forest' == technique:
		#monitor_name = technique+'_not_optimized'
		monitor_name = technique+'_optimized'
		monitor = create_monitor(technique, dataset_name, monitor_name, monitoring_characteristics)
		
		return np.array([monitor])


def load_box_based_monitors(dataset_name, technique, classes_to_monitor, params):

	monitors = []
	arr_n_clusters_oob = params['arr_n_clusters']
	
	for n_clusters_oob in arr_n_clusters_oob:
		monitor = None
		boxes = {}

		if 'oob' == technique:
			monitor_name = technique+'_{}_clusters'.format(n_clusters_oob)
			monitor = Monitor(monitor_name)

			#for class_to_monitor in range(classes_to_monitor):
			monitor_folder = root_path +sep+ monitoring_characteristics +sep+ dataset_name +sep
			#	monitor_folder += technique +sep+ 'class_'+str(class_to_monitor) +sep
			monitor_folder += technique +sep+ 'class_'
			monitor.monitors_folder = monitor_folder
			#monitor.filename = 'monitor_'+monitor_name+'.p'
			monitor.filename = 'monitor_'+monitor_name+'.p_2' #built with true labels instead of right predictions
				#monitor_path = monitor.monitors_folder+monitor.filename
				# loading abstraction boxes
				#boxes[class_to_monitor] = pickle.load(open(monitor_path, "rb"))

			if 'ensemble' in technique:
				monitor.method = abstraction_box.find_point_box_ensemble
			else:
				monitor.method = abstraction_box.find_point

			monitor.dim_reduc_method = None
			#monitor.boxes = boxes
			monitors.append(monitor)
			
		elif 'oob_isomap' == technique or 'oob_pca' == technique or 'oob_pca_isomap' == technique:
			arr_n_components = params['arr_n_components']

			for n_components in arr_n_components:
				boxes = {}
				dim_reduc_method = []
				
				monitor_name = technique+'_{}_components_{}_clusters'.format(n_components, n_clusters_oob)
				reduc_name = technique+'_{}_components'.format(n_components)

				monitor = Monitor(monitor_name)

				#for class_to_monitor in range(classes_to_monitor):
				monitor_folder = root_path +sep+ monitoring_characteristics +sep+ dataset_name +sep
				#	monitor_folder += technique +sep+ 'class_'+str(class_to_monitor) +sep
				monitor_folder += technique +sep+ 'class_'
				monitor.monitors_folder = monitor_folder
				monitor.filename = 'monitor_'+monitor_name+'.p'
				#monitor.filename = 'monitor_'+monitor_name+'.p_2' #built with true labels instead of right predictions
				#	dim_reduc_method[class_to_monitor] = pickle.load(open(monitor_folder+'trained_'+reduc_name+'.p', "rb"))
				
					# loading abstraction boxes
					#monitor_path = monitor.monitors_folder+monitor.filename
					#boxes[class_to_monitor] = pickle.load(open(monitor_path, "rb"))
				
				if 'ensemble' in technique:
					monitor.method = abstraction_box.find_point_box_ensemble
				else:
					monitor.method = abstraction_box.find_point

				#monitor.dim_reduc_method = dim_reduc_method
				monitor.dim_reduc_method = reduc_name

				if 'oob_pca_isomap' == technique:
					dim_reduc_method.append('PCA_'+reduc_name+'.p')
					dim_reduc_method.append('Isomap_'+reduc_name+'.p')
					monitor.dim_reduc_method = dim_reduc_method

				#monitor.boxes = boxes
				monitors.append(monitor)

		elif '' == technique:
			pass

	return np.array(monitors)