import numpy as np
from src.utils import util
from src.Classes.monitor import Monitor
from src.novelty_detection.methods import abstraction_box
from src.novelty_detection.methods import act_func_based_monitor
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


sep = util.get_separator()
experiment_type = 'novelty_detection'
root_path = 'src'+sep+experiment_type+sep+'bin'+sep+'monitors'


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


def prepare_box_based_monitors(dataset_name, technique_names, class_to_monitor,
 arr_n_clusters_oob = [3], arr_n_components = [2]):

	monitoring_characteristics = 'dnn_internals'
	monitors = []
	c = '{}'.format(class_to_monitor)
	for technique in technique_names:
		monitor_folder = root_path +sep+ monitoring_characteristics +sep+ dataset_name +sep
		monitor_folder += technique +sep+ 'class_'+ c +sep

		for n_clusters_oob in arr_n_clusters_oob:

			if 'oob' == technique:
				monitor_name = technique+'_{}_clusters'.format(n_clusters_oob)
				
				monitor = build_abstraction_based_monitor(class_to_monitor, monitor_name, n_clusters_oob, monitor_folder)

				monitors.append(monitor)

			elif 'oob_gradient' == technique:
				monitor_name = technique+'_{}_clusters'.format(n_clusters_oob)
				
				monitor = build_gradient_based_monitor(class_to_monitor, monitor_name, n_clusters_oob, monitor_folder)

				monitors.append(monitor)
			
			elif 'oob_isomap' == technique or 'oob_pca' == technique or 'oob_pca_isomap' == technique:
				
				dim_reduc_method = []
				dim_reduc_filename_prefix = []

				for n_components in arr_n_components:
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
					
					monitors.append(monitor)

			elif '' == technique:
				pass

	return monitors