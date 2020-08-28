from src import model_config
from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.novelty_detection.evaluators import dnn_oob_evaluator
from src.novelty_detection.testers import dnn_oob_tester
from src.novelty_detection.testers import en_dnn_oob_tester
from src.novelty_detection.methods import abstraction_box
from src.novelty_detection.methods import act_func_based_monitor
from src.utils import util


sep = util.get_separator()

def load_file_names():
	index_file =  'index_mapping_results.csv'
	acc_file_name = 'accuracies.csv'
	cf_file_name = 'positive_negative_rates.csv'
	time_file_name = 'time.csv'
	mem_file_name = 'memory.csv'
	f1_file_name = 'f1.csv'

	return [index_file, acc_file_name, cf_file_name, time_file_name, mem_file_name, f1_file_name]


def load_settings(monitor_acronym, dataset):
	monitor = None
	if dataset == 'MNIST':
		monitor = Monitor('novelty_detection', load_var("e1_d1_mn"))
	elif dataset == 'GTSRB':
		monitor = Monitor('novelty_detection', load_var("e1_d2_mn"))
	
	if monitor_acronym == 'oob':
		monitor.trainer = act_func_based_monitor
		monitor.method = abstraction_box.make_abstraction
		monitor.n_clusters = 3
		
	elif monitor_acronym == 'oob_isomap':
		monitor.trainer = act_func_based_monitor
		monitor.method = abstraction_box.make_abstraction
		monitor.n_clusters = 3
		monitor.dim_reduc_method = 'isomap'
		monitor.file_dim_reduc_method = monitor.dim_reduc_method+'_'+dataset
		monitor.n_components = 2

	return monitor




'''
elif experiment_number == 2:
	experiment = Experiment('DNN_OOB_NL')
	
	for monitor in monitors:
		monitor.dim_reduc_method = load_var('e'+str(experiment_number)+'_d'+str(i+1)+'_dr')

elif experiment_number == 3:
	#Experiment 3: Same of 1 but using Ensemble of DNN
	experiment = Experiment('ENSBL_OOB')
	#models
	dnn_mnist = ModelBuilder()
	dnn_mnist.model_name = var_dict['e3_d1_md']
	dnn_mnist.num_cnn = 3
	dnn_gtsrb = ModelBuilder()
	dnn_gtsrb.model_name = var_dict['e3_d2_md']
	dnn_gtsrb.num_cnn = 5
	modelsObj = [dnn_mnist, dnn_gtsrb]
	#monitors
	monitorObjMNIST = Monitor(var_dict['e3_d1_mn'], load_var_dict["classToMonitor"], load_var_dict["layerToMonitor"])
	monitorObjMNIST.method = abstraction_box.find_point_box_ensemble
	monitorObjMNIST.monitors_folder += var_dict['e3_folder'] + sep
	monitorObjGTSRB = Monitor(var_dict['e3_d2_mn'], load_var_dict["classToMonitor"], load_var_dict["layerToMonitor"])
	monitorObjGTSRB.method = abstraction_box.find_point_box_ensemble
	monitorObjGTSRB.monitors_folder += var_dict['e3_folder'] + sep
	monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
	#building the class experiment 3
	experiment.models = modelsObj
	experiment.monitors = monitorsObj
	experiment.evaluator = en_dnn_oob_evaluator
	experiment.tester = en_dnn_oob_tester

	return experiment

elif experiment_number == 4:
	#Experiment 4: Ensemble of DNN with outside-of-box monitor and dimensionality reduction method
	experiment = Experiment('ENSBL+OB+NL')
	#using the same ML models from the Experiment 3
	dnn_mnist = ModelBuilder()
	dnn_mnist.model_name = var_dict['e4_d1_md']

	dnn_gtsrb = ModelBuilder()
	dnn_gtsrb.model_name = var_dict['e4_d2_md']
	
	modelsObj = [dnn_mnist, dnn_gtsrb]
	#monitors
	monitorObjMNIST = Monitor(var_dict['e4_d1_mn'], load_var_dict["classToMonitor"], load_var_dict["layerToMonitor"])
	monitorObjMNIST.method = abstraction_box.find_point_box_ensemble
	monitorObjMNIST.dim_reduc_method = var_dict['e4_d1_dr']
	monitorObjMNIST.monitors_folder += var_dict['e4_folder'] + sep

	monitorObjGTSRB = Monitor(var_dict['e4_d2_mn'], load_var_dict["classToMonitor"], load_var_dict["layerToMonitor"])
	monitorObjGTSRB.method = abstraction_box.find_point_box_ensemble
	monitorObjGTSRB.dim_reduc_method = var_dict['e4_d2_dr']
	monitorObjGTSRB.monitors_folder += var_dict['e4_folder'] + sep

	monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
	#building the class experiment 4
	experiment.models = modelsObj
	experiment.monitors = monitorsObj
	experiment.evaluator = en_dnn_oob_evaluator
	experiment.tester = en_dnn_oob_tester

	return experiment

	#building the class experiment
	experiment.models = models
	experiment.monitors = monitors
	experiment.tester = dnn_oob_tester
	experiment.evaluator = dnn_oob_evaluator

	return experiment
'''


'''
**OK** 1 = outside-of-box paper; 2 = outside-of-box using isomap instead of 2D projection;

**testing** 3 = outside-of-box with ensemble of DNN; 4 = same of 3 but using isomap strategy;

5 = same of 2 but using DBSCAN instead of KNN; 6 = same of 2 but clustering without dimension reduction;
7 = same of 5 but clustering without dimension reduction; 
8 = using the derivative of activation functions instead of raw values
'''

def load_vars(experiment_type, key):
	var_dict = {}

	var_dict['validation_size'] = 0.3
	
	var_dict['model_names'] = ['leNet', 'leNet']#, 'leNet']	
	var_dict['batch_numbers'] = [128, 10]
	var_dict['epoch_numbers'] = [12, 32]
	
	# monitors
	if experiment_type == 'novelty_detection':
		var_dict['technique_names'] = ['oob', 'oob_isomap', 'oob_pca']#'oob_gradient'

	return var_dict[key]