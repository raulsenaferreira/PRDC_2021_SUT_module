import os
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
from src.utils import metrics
from pathos.multiprocessing import ProcessingPool as Pool
from src.novelty_detection import config as config_ND
from src.Classes.dataset import Dataset
from keras.models import load_model
from sklearn import manifold
import pickle


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


def save_results(experiment, arr_readouts, plot=False):
	filenames = config_ND.load_file_names()
	csvs_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'csv'+sep+experiment.experiment_type+sep+experiment.name+sep
	img_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'img'+sep+experiment.experiment_type+sep+experiment.name+sep

	metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(experiment.name, arr_readouts, img_folder_path+'all_images.pdf')


if __name__ == "__main__":
	## settings
	experiments_pool = []

	# variables regarding Novelty-Detection runtime-monitoring experiments
	experiment_type = 'novelty_detection'
	dataset_names = config_ND.load_vars(experiment_type, 'dataset_names')
	validation_size = config_ND.load_vars(experiment_type, 'validation_size')
	model_names = config_ND.load_vars(experiment_type, 'model_names')
	monitor_names = config_ND.load_vars(experiment_type, 'monitor_names')
	classes_to_monitor = config_ND.load_vars(experiment_type, 'classes_to_monitor')
	arr_n_components_isomap = [2, 3, 5, 10]#n_components_isomap = 2 #fixo por enquanto, tranformar em array depois

	# other settings
	parallel_execution = True
	repetitions = 1
	perc_of_data = 0.05 #e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data

	## loading experiments
	for model_name, dataset_name, class_to_monitor in zip(model_names, dataset_names, classes_to_monitor):
		monitors = []
		# loading dataset
		dataset = Dataset(dataset_name)

		# loading model
		model = ModelBuilder()
		#print(model.models_folder+model_name+'_'+dataset_name+'.h5')
		model = load_model(model.models_folder+model_name+'_'+dataset_name+'.h5')

		# *** Inicio funcao que contem a logica de criacao dos monitores
		# loading monitors
		for monitor_name in monitor_names:
			monitor = Monitor(monitor_name, experiment_type)
			monitor.class_to_monitor = class_to_monitor
			
			## diferent way for inspecting a decision, if ensemble or standalone model
			if 'ensemble' in model_name:
				monitor.method = abstraction_box.find_point_box_ensemble
			else:
				monitor.method = abstraction_box.find_point

			if 'isomap' in monitor_name:
				for n_components_isomap in arr_n_components_isomap:
					file_isomap = 'isomap_' + str(n_components_isomap) + '_components_' + dataset_name + '_class_' + str(monitor.class_to_monitor) + '.p'
					monitor.dim_reduc_method = pickle.load(open(monitor.monitors_folder+file_isomap, "rb"))
					monitors.append(monitor)
		# *** Fim funcao que contem a logica de criacao dos monitores
			else:
				monitors.append(monitor)

		# creating an instance of an experiment
		experiment = Experiment(model_name+'_'+dataset_name+'_experiments')
		experiment.experiment_type = experiment_type
		experiment.dataset = dataset
		experiment.model = model
		experiment.monitors = monitors

		## diferent evaluator and tester, if ensemble or standalone model
		if 'ensemble' in model_name:
			experiment.evaluator = en_dnn_oob_evaluator
			experiment.tester = en_dnn_oob_tester
		else:
			experiment.tester = dnn_oob_tester
			experiment.evaluator = dnn_oob_evaluator

		experiments_pool.append(experiment)

	#Running experiments: parallel or not
	if parallel_execution:
		#Parallelizing the experiments (optional): one experiment per Core
		pool = Pool()
		processes_pool = []

		for experiment in experiments_pool:
			processes_pool.append(pool.apipe(experiment.evaluator.evaluate, repetitions, experiment, perc_of_data))  
		
		timeout = 1800 * len(experiment.monitors)
		print("Max seconds to run:", timeout)

		for process in processes_pool:
			arr_readouts = process.get(timeout=timeout)	
			save_results(experiment, arr_readouts, plot=True)
	else:
		#Serial version for the experiments
		for experiment in experiments_pool:
			arr_readouts = experiment.evaluator.evaluate(repetitions, experiment, perc_of_data) 
			save_results(experiment, arr_readouts, plot=True)

		#Experiment 4 SCGAN with MNIST for novelty/OOD detection
		#model_name = 'DNN_MNIST.h5'
		#monitor_name = 'SCGAN_MNIST__3.h5'
		#monitors_folder = monitors_folder+'SCGAN_checkpoint'+sep
		#arrPred, count, arrFP, arrFN, arrTP, arrTN = SCGAN_MNIST_test.run(classToMonitor, models_folder, 
		#	monitors_folder, model_name, monitor_name)