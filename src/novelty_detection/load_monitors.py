import os 
from src.utils import util
from sklearn import manifold
from src.Classes.monitor import Monitor
from src.novelty_detection.methods import abstraction_box
from src.novelty_detection.methods import act_func_based_monitor
from sklearn.decomposition import PCA
import pickle
import numpy as np


sep = util.get_separator()
dir_path = os.path.dirname(os.path.realpath(__file__))
experiment_type = 'novelty_detection'
#root_path = 'src'+sep+experiment_type+sep+'bin'+sep+'monitors'
root_path = dir_path+sep+'bin'+sep+'monitors'


#cwd = os.getcwd()
#print(dir_path)
#print(cwd)

def load_box_based_monitors(dataset_name, technique, classes_to_monitor,
 arr_n_clusters_oob = [3], arr_n_components = [2]):

	monitoring_characteristics = 'dnn_internals'
	monitors = []
	
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
			monitor.filename = 'monitor_'+monitor_name+'.p'
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
			
		elif 'oob_isomap' == technique or 'oob_pca' == technique:
			
			for n_components in arr_n_components:
				boxes = {}
				dim_reduc_method = {}
				
				monitor_name = technique+'_{}_components_{}_clusters'.format(n_components, n_clusters_oob)
				reduc_name = technique+'_{}_components'.format(n_components)

				monitor = Monitor(monitor_name)

				#for class_to_monitor in range(classes_to_monitor):
				monitor_folder = root_path +sep+ monitoring_characteristics +sep+ dataset_name +sep
				#	monitor_folder += technique +sep+ 'class_'+str(class_to_monitor) +sep
				monitor_folder += technique +sep+ 'class_'
				monitor.monitors_folder = monitor_folder
				monitor.filename = 'monitor_'+monitor_name+'.p'
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
				#monitor.boxes = boxes
				monitors.append(monitor)

		elif '' == technique:
			pass

	return np.array(monitors)