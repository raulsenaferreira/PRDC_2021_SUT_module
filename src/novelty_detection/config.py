from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.novelty_detection import dnn_oob_evaluator
from src.novelty_detection import dnn_oob_tester
from src.novelty_detection import en_dnn_oob_tester
from src.novelty_detection.utils import abstraction_box
from src.utils import util


sep = util.get_separator()

def load_file_names(experiment_type='novelty_detection'):
	if experiment_type == 'novelty_detection' or experiment_type == 'distributional_shift' or experiment_type == 'adversarial_attack':
		compiled_img_name = experiment_type+sep+'all_images.pdf'
		acc_file_name = experiment_type+sep+'accuracies.csv'
		cf_file_name = experiment_type+sep+'positive_negative_rates.csv'
		time_file_name = experiment_type+sep+'time.csv'
		mem_file_name = experiment_type+sep+'memory.csv'
		f1_file_name = experiment_type+sep+'f1.csv'

		return compiled_img_name, acc_file_name, cf_file_name, time_file_name, mem_file_name, f1_file_name


def load_experiment(experiment_number):
	classToMonitor = 7
	layerToMonitor = -2
	
	if experiment_number == 1:
		#Experiment 1: DNN with outside-of-box monitor
		experiment = Experiment('DNN+OB')
		#models
		dnn_mnist = ModelBuilder()
		dnn_mnist.model_name = 'DNN_MNIST.h5'
		dnn_mnist.exec = dnn_oob_tester
		dnn_gtsrb = ModelBuilder()
		dnn_gtsrb.model_name = 'DNN_GTRSB.h5'
		dnn_gtsrb.exec = dnn_oob_tester
		modelsObj = [dnn_mnist, dnn_gtsrb]
		#monitors
		monitorObjMNIST = Monitor("monitor_Box_MNIST_class_{}.p".format(classToMonitor), classToMonitor, layerToMonitor)
		monitorObjMNIST.method = abstraction_box.find_point
		monitorObjGTSRB = Monitor("monitor_Box_GTRSB_class_{}.p".format(classToMonitor), classToMonitor, layerToMonitor)
		monitorObjGTSRB.method = abstraction_box.find_point
		monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
		#building the class experiment 1
		experiment.models = modelsObj
		experiment.monitors = monitorsObj
		experiment.evaluator = dnn_oob_evaluator

		return experiment

	elif experiment_number == 2:
		#Experiment 2: DNN with outside-of-box monitor using non-linear dimensionality reduction
		experiment = Experiment('DNN+OB+NL')
		#using the same ML models from the Experiment 1
		dnn_mnist = ModelBuilder()
		dnn_mnist.model_name = 'DNN_MNIST.h5'
		dnn_mnist.exec = dnn_oob_tester
		dnn_gtsrb = ModelBuilder()
		dnn_gtsrb.model_name = 'DNN_GTRSB.h5'
		dnn_gtsrb.exec = dnn_oob_tester
		modelsObj = [dnn_mnist, dnn_gtsrb]
		#monitors
		monitorObjMNIST = Monitor("monitor_Box_isomap_MNIST_class_{}.p".format(classToMonitor), classToMonitor, layerToMonitor)
		monitorObjMNIST.method = abstraction_box.find_point
		monitorObjMNIST.dim_reduc_method = 'isomap_MNIST_trained_class_{}.p'.format(classToMonitor)
		monitorObjGTSRB = Monitor("monitor_Box_isomap_GTRSB_class_{}.p".format(classToMonitor), classToMonitor, layerToMonitor)
		monitorObjGTSRB.method = abstraction_box.find_point
		monitorObjGTSRB.dim_reduc_method = 'isomap_GTSRB_trained_class_{}.p'.format(classToMonitor)
		monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
		#building the class experiment 2
		experiment.models = modelsObj
		experiment.monitors = monitorsObj
		experiment.evaluator = dnn_oob_evaluator

		return experiment

	elif experiment_number == 3:
		#Experiment 3: Ensemble of DNN with outside-of-box monitor
		experiment = Experiment('ENSBL+OB')
		#models
		dnn_mnist = ModelBuilder()
		dnn_mnist.model_name = 'DNN_ensemble_MNIST_'
		dnn_mnist.exec = en_dnn_oob_tester
		dnn_mnist.num_cnn = 3
		dnn_gtsrb = ModelBuilder()
		dnn_gtsrb.model_name = 'DNN_ensemble_GTRSB_'
		dnn_gtsrb.exec = en_dnn_oob_tester
		dnn_gtsrb.num_cnn = 5
		modelsObj = [dnn_mnist, dnn_gtsrb]
		#monitors
		monitorObjMNIST = Monitor("monitor_Box_MNIST_class_{}.p".format(classToMonitor), classToMonitor, layerToMonitor)
		monitorObjMNIST.method = abstraction_box.find_point_box_ensemble
		monitorObjMNIST.monitors_folder += 'outOfBox_ensembleDNN' + sep
		monitorObjGTSRB = Monitor("outOfBox_GTRSB_class_{}.p".format(classToMonitor), classToMonitor, layerToMonitor)
		monitorObjGTSRB.method = abstraction_box.find_point_box_ensemble
		monitorObjGTSRB.monitors_folder += 'outOfBox_ensembleDNN' + sep
		monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
		#building the class experiment 3
		experiment.models = modelsObj
		experiment.monitors = monitorsObj
		experiment.evaluator = dnn_oob_evaluator

		return experiment

	elif experiment_number == 4:
		#Experiment 4: Ensemble of DNN with outside-of-box monitor and dimensionality reduction method
		experiment = Experiment('ENSBL+OB+NL')
		#using the same ML models from the Experiment 3
		dnn_mnist = ModelBuilder()
		dnn_mnist.model_name = 'DNN_ensemble_MNIST_'
		dnn_mnist.exec = dnn_oob_tester
		dnn_gtsrb = ModelBuilder()
		dnn_gtsrb.model_name = 'DNN_ensemble_GTRSB_'
		dnn_gtsrb.exec = dnn_oob_tester
		modelsObj = [dnn_mnist, dnn_gtsrb]
		#monitors
		monitorObjMNIST = Monitor("monitor_Box_isomap_MNIST_class_{}.p".format(classToMonitor), classToMonitor, layerToMonitor)
		monitorObjMNIST.method = abstraction_box.find_point_box_ensemble
		monitorObjMNIST.dim_reduc_method = 'isomap_MNIST_trained_class_{}.p'.format(classToMonitor)
		monitorObjGTSRB = Monitor("monitor_Box_isomap_GTRSBclass_{}.p".format(classToMonitor), classToMonitor, layerToMonitor)
		monitorObjGTSRB.method = abstraction_box.find_point_box_ensemble
		monitorObjGTSRB.dim_reduc_method = 'isomap_GTSRB_trained_class_{}.p'.format(classToMonitor)
		monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
		#building the class experiment 4
		experiment.models = modelsObj
		experiment.monitors = monitorsObj
		experiment.evaluator = dnn_oob_evaluator

		return experiment