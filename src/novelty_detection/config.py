from src import model_config
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.novelty_detection import dnn_oob_evaluator
from src.novelty_detection import dnn_oob_tester
from src.novelty_detection import en_dnn_oob_tester
from src.novelty_detection.utils import abstraction_box
from src.MNIST_experiments import act_func_based_monitor
from src.utils import util


sep = util.get_separator()

def load_file_names():
	experiment_type = 'novelty_detection'
	compiled_img_name = experiment_type+sep+'all_images.pdf'
	acc_file_name = experiment_type+sep+'accuracies.csv'
	cf_file_name = experiment_type+sep+'positive_negative_rates.csv'
	time_file_name = experiment_type+sep+'time.csv'
	mem_file_name = experiment_type+sep+'memory.csv'
	f1_file_name = experiment_type+sep+'f1.csv'

	return compiled_img_name, acc_file_name, cf_file_name, time_file_name, mem_file_name, f1_file_name


def load_settings(monitor_acronym):
	monitor = None
	monitors_folder = load_var('monitors_folder')
	if monitor_acronym == 'oob':
		monitor = Monitor(load_var("e1_d1_mn"), load_var("classToMonitor"), load_var("layerToMonitor"))
		monitor.method = abstraction_box.make_abstraction
		monitor.trainer = act_func_based_monitor
		monitor.monitors_folder = monitors_folder

	return monitor


def load_experiment_settings(experiment_number):
		
	if experiment_number == 1:
		#Experiment 1: DNN with outside-of-box monitor
		experiment = Experiment('DNN+OB')

		#models
		dnn_mnist = model_config.load_model_settings(1)
		dnn_gtsrb = model_config.load_model_settings(2)
		modelsObj = [dnn_mnist, dnn_gtsrb]
		
		#monitors
		monitorObjMNIST = Monitor(load_var_dict["e1_d1_mn"], load_var_dict["classToMonitor"], load_var_dict["layerToMonitor"])
		monitorObjMNIST.method = abstraction_box.find_point

		monitorObjGTSRB = Monitor(load_var_dict["e1_d2_mn"], load_var_dict["classToMonitor"], load_var_dict["layerToMonitor"])
		monitorObjGTSRB.method = abstraction_box.find_point
		monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
		
		#building the class experiment 1
		experiment.models = modelsObj
		experiment.monitors = monitorsObj
		experiment.tester = dnn_oob_tester
		experiment.evaluator = dnn_oob_evaluator

		return experiment

	elif experiment_number == 2:
		#Experiment 2: DNN with outside-of-box monitor using non-linear dimensionality reduction
		experiment = Experiment('DNN+OB+NL')
		#using the same ML models from the Experiment 1
		dnn_mnist = ModelBuilder()
		dnn_mnist.model_name = load_var_dict["e2_d1_md"]
		
		dnn_gtsrb = ModelBuilder()
		dnn_gtsrb.model_name = var_dict['e2_d2_md']

		modelsObj = [dnn_mnist, dnn_gtsrb]
		
		#monitors
		monitorObjMNIST = Monitor(var_dict['e2_d1_mn'], load_var_dict["classToMonitor"], load_var_dict["layerToMonitor"])
		monitorObjMNIST.method = abstraction_box.find_point
		monitorObjMNIST.dim_reduc_method = var_dict['e2_d1_dr']
		monitorObjGTSRB = Monitor(var_dict['e2_d2_mn'], load_var_dict["classToMonitor"], load_var_dict["layerToMonitor"])
		monitorObjGTSRB.method = abstraction_box.find_point
		monitorObjGTSRB.dim_reduc_method = var_dict['e2_d2_dr']
		monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
		
		#building the class experiment 2
		experiment.models = modelsObj
		experiment.monitors = monitorsObj
		experiment.evaluator = dnn_oob_evaluator
		experiment.tester = dnn_oob_tester

		return experiment

	elif experiment_number == 3:
		#Experiment 3: Ensemble of DNN with outside-of-box monitor
		experiment = Experiment('ENSBL+OB')
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


def load_var(key):
	''' 
	Change here the values of the variables if you want to change:
	1) the DNN layer and/or class monitored by the runtime monitor 
	2) the name of the generated files

	Logic for global variable mapping (variables that are used for build model/monitor files and to rn tests)
	e1 = Experiment 1 (order of the experiments in this code)
	d1 = dataset 1 (MNIST = 1; GTSRB = 2; CIFAR-10 = 3 ...)
	mn = monitor; md = model; dr = dimensionality reduction

	Example for the monitor name for the experiment 1 + dataset 1 --> "e1_d1_mn"
	'''
	var_dict = {}
	classToMonitor = 7
	var_dict['classToMonitor'] = classToMonitor
	layerToMonitor = -2
	var_dict['layerToMonitor'] = layerToMonitor

	var_dict['monitors_folder'] = "src"+sep+"novelty_detection"+sep+"bin"+sep+"monitors"+sep

	var_dict['m1_d1_name'] = 'DNN_MNIST.h5'
	var_dict['m1_d1_batch'] = 128
	var_dict['m1_d1_epoch'] = 12
	var_dict['m1_d1_name'] = 'DNN_GTRSB.h5'
	var_dict['m1_d2_batch'] = 10
	var_dict['m1_d2_epoch'] = 32
	var_dict['m2_d1_name'] = 'DNN_ensemble_MNIST_'
	var_dict['m2_d2_name'] = 'DNN_ensemble_GTRSB_'

	var_dict['e1_d1_md'] = 'DNN_MNIST.h5'
	var_dict['e1_d2_md'] = 'DNN_GTRSB.h5'
	var_dict['e1_d1_mn'] = "outOfBox_MNIST_class_{}.p".format(classToMonitor)
	var_dict['e1_d2_mn'] = "outOfBox_GTRSB_class_{}.p".format(classToMonitor)
	
	var_dict['e2_d1_md'] = 'DNN_MNIST.h5'
	var_dict['e2_d2_md'] = 'DNN_GTRSB.h5'
	var_dict['e2_d1_mn'] = "outOfBox_isomap_MNIST_class_{}.p".format(classToMonitor)
	var_dict['e2_d2_mn'] = "outOfBox_isomap_GTRSB_class_{}.p".format(classToMonitor)
	var_dict['e2_d1_dr'] = 'isomap_MNIST_trained_class_{}.p'.format(classToMonitor)
	var_dict['e2_d2_dr'] = 'isomap_GTSRB_trained_class_{}.p'.format(classToMonitor)

	var_dict['e3_d1_md'] = 'DNN_ensemble_MNIST_'
	var_dict['e3_d2_md'] = 'DNN_ensemble_GTRSB_'
	var_dict['e3_d1_mn'] = "outOfBox_MNIST_class_{}.p".format(classToMonitor)
	var_dict['e3_d2_mn'] = "outOfBox_GTRSB_class_{}.p".format(classToMonitor)
	var_dict['e3_folder'] = 'outOfBox_ensembleDNN'

	var_dict['e4_d1_md'] = 'DNN_ensemble_MNIST_'
	var_dict['e4_d2_md'] = 'DNN_ensemble_GTRSB_'
	var_dict['e4_d1_mn'] = "outOfBox_isomap_MNIST_class_{}.p".format(classToMonitor)
	var_dict['e4_d2_mn'] = "outOfBox_isomap_GTRSB_class_{}.p".format(classToMonitor)
	var_dict['e4_d1_dr'] = 'isomap_MNIST_trained_class_{}.p'.format(classToMonitor)
	var_dict['e4_d2_dr'] = 'isomap_GTSRB_trained_class_{}.p'.format(classToMonitor)
	var_dict['e4_folder'] = 'outOfBox_NL_ensembleDNN'

	return var_dict[key]