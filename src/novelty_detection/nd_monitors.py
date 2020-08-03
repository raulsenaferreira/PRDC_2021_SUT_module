from sklearn import manifold
from src.Classes.monitor import Monitor
from src.novelty_detection.methods import abstraction_box
from src.novelty_detection.methods import act_func_based_monitor


def build_monitors(dataset_name, monitor_names, class_to_monitor, n_clusters_oob = 3, n_components_isomap = 2):
	experiment_type = 'novelty_detection'
	monitors = []

	for monitor_name in monitor_names:
		monitor = Monitor(monitor_name, experiment_type)
		monitor.class_to_monitor = class_to_monitor

		if 'oob' in monitor_name:
			monitor.trainer = act_func_based_monitor
			monitor.method = abstraction_box.make_abstraction
			monitor.n_clusters = n_clusters_oob
			monitor.dim_reduc_method = None
		if 'isomap' in monitor_name:
			monitor.dim_reduc_method = manifold.Isomap(n_components = n_components_isomap)
			monitor.dim_reduc_filename_prefix = 'isomap_' + str(n_components_isomap) + '_components_' + dataset_name + '_class_' + str(monitor.class_to_monitor) + '.p'

		monitors.append(monitor)

	return monitors