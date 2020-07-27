#from src.GTRSB_experiments import DNN_outOfBox_dimReduc_test
#from src.GTRSB_experiments import DNN_ensemble_outOfBox_GTRSB_test
#from src.GTRSB_experiments import DNN_outOfBox_GTRSB_test
#from src.MNIST_experiments import SCGAN_MNIST_test
from src import model_config
from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.Classes.experiment import Experiment
from src.novelty_detection import dnn_oob_evaluator
from src.novelty_detection import dnn_oob_tester
from src.novelty_detection import en_dnn_oob_tester
from src.novelty_detection.utils import abstraction_box
from src.Classes import act_func_based_monitor
from src.utils import util
from src.utils import metrics
#from src import cnn_nl_oob_experiments
from pathos.multiprocessing import ProcessingPool as Pool
from src.novelty_detection import config as config_ND
from src.Classes.dataset import Dataset
from keras.models import load_model


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


if __name__ == "__main__":
	
	## global settings
	experiment_type = 'novelty_detection'
	experiments_pool = []
	sep = util.get_separator()
	repetitions = 1
	model_names = ['leNet', 'leNet']
	dataset_names = ['MNIST', 'GTSRB']
	#each indice corresponds a dataset, each value corresponds a class to monitor in this dataset
	classesToMonitor = [1, 7] 
	monitor_names = ['oob', 'oob_isomap']
	
	parallel_execution = False
	timeout = 900 * len(dataset_names) #suggesting 15 min for performing parallel experiment on each dataset
	
	csvs_folder_path = 'results'+sep+'csv'+sep
	img_folder_path = 'results'+sep+'img'+sep

	# variables regarding the results for Novelty-Detection runtime-monitoring experiments
	compiled_img_name, acc_file_name, cf_file_name, time_file_name, mem_file_name, f1_file_name = config_ND.load_file_names(experiment_type)
	accuracies = []
	confusion_matrices = []
	proc_times = []
	memories = []
	f1s = []

	## loading experiments
	for model_name, dataset_name, classToMonitor in zip(model_names, dataset_names, classesToMonitor):
		monitors = []

		# loading dataset
		dataset = Dataset(dataset_name)

		# loading model
		model = ModelBuilder()
		model_file_name = config_ND.load_novelty_detection_vars(model_name+'_model')
		model.binary = load_model(model.models_folder+model_file_name)

		# *** Inicio funcao que contem a logica de criacao dos monitores
		# loading monitors
		for monitor_name in monitor_names:
			monitor_file_name = model_name+'_monitor_'+monitor_name
			monitor = Monitor(experiment_type, config_ND.load_novelty_detection_vars(monitor_file_name))
			monitor.classToMonitor = classToMonitor
			
			## diferent way for inspecting a decision, if ensemble or standalone model
			if 'ensemble' in model_name:
				monitor.method = abstraction_box.find_point_box_ensemble
			else:
				monitor.method = abstraction_box.find_point

			if 'isomap' in monitor_name:
				monitor.dim_reduc_method = monitor_file_name+"_trained"
			else:
				monitor.dim_reduc_method = None
		# *** Fim funcao que contem a logica de criacao dos monitores

			monitors.append(monitor)

		# creating an instance of an experiment
		experiment = Experiment(model_name+'_experiments')
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
			processes_pool.append(pool.apipe(experiment.evaluator.evaluate, 
				repetitions, experiment.name, experiment.models, experiment.datasets, experiment.monitors))  
		
		for process in processes_pool:
			avg_acc, avg_time, avg_cf, avg_mem, avg_f1 = process.get(timeout=timeout)

			accuracies.append(avg_acc)
			confusion_matrices.append(avg_cf)
			proc_times.append(avg_time)
			memories.append(avg_mem)
			f1s.append(avg_f1)		
	else:
		#Serial version for the experiments
		for experiment in experiments_pool:
			avg_acc, avg_time, avg_cf, avg_mem, avg_f1 = experiment.evaluator.evaluate(repetitions, experiment) 
			
			accuracies.append(avg_acc)
			confusion_matrices.append(avg_time)
			proc_times.append(avg_cf)
			memories.append(avg_mem)
			f1s.append(avg_f1)

		#Experiment 4 SCGAN with MNIST for novelty/OOD detection
		#model_name = 'DNN_MNIST.h5'
		#monitor_name = 'SCGAN_MNIST__3.h5'
		#monitors_folder = monitors_folder+'SCGAN_checkpoint'+sep
		#arrPred, count, arrFP, arrFN, arrTP, arrTN = SCGAN_MNIST_test.run(classToMonitor, models_folder, 
		#	monitors_folder, model_name, monitor_name)

	# CSVs
	metrics.save_results(accuracies, csvs_folder_path+acc_file_name, ',')
	metrics.save_results(confusion_matrices, csvs_folder_path+cf_file_name, ',')
	metrics.save_results(proc_times, csvs_folder_path+time_file_name, ',')
	metrics.save_results(memories, csvs_folder_path+mem_file_name, ',')
	metrics.save_results(f1s, csvs_folder_path+f1_file_name, ',')
	# Figures
	metrics.plot_pos_neg_rate_stacked_bars(confusion_matrices, dataset_names, img_folder_path+compiled_img_name)