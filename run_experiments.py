#from src.GTRSB_experiments import DNN_outOfBox_dimReduc_test
#from src.GTRSB_experiments import DNN_ensemble_outOfBox_GTRSB_test
#from src.GTRSB_experiments import DNN_outOfBox_GTRSB_test
#from src.MNIST_experiments import SCGAN_MNIST_test
from src.utils import util
from src.utils import metrics
#from src import cnn_nl_oob_experiments
from pathos.multiprocessing import ProcessingPool as Pool
from src.novelty_detection import config as config_ND
from src.Classes.dataset import Dataset


# ML is incorrect but monitor does not trust on it = TP
# ML is correct but monitor does not trust on it = FP
# ML is incorrect and monitor trust on it = FN
# ML is correct and monitor trust on it = TN


if __name__ == "__main__":
	
	#global settings
	experiments_pool = []
	sep = util.get_separator()
	repetitions = 1
	dataset_names = ['MNIST', 'GTSRB']
	
	parallel_execution = False
	timeout = 900 * len(dataset_names) #suggesting 15 min for performing parallel experiment on each dataset
	
	csvs_folder_path = 'results'+sep+'csv'+sep
	img_folder_path = 'results'+sep+'img'+sep
	experiment_type = ['novelty_detection', 'distributional_shift', 'adversarial_attack']

	#datasets
	mnistObj = Dataset(dataset_names[0]) 
	gtsrbObj = Dataset(dataset_names[1]) 
	datasetObjs = [mnistObj, gtsrbObj]
	
	# variables regarding the results for Novelty-Detection runtime-monitoring experiments
	compiled_img_name, acc_file_name, cf_file_name, time_file_name, mem_file_name, f1_file_name = config_ND.load_file_names(experiment_type[0])
	accuracies = []
	confusion_matrices = []
	proc_times = []
	memories = []
	f1s = []

	experiment = config_ND.load_experiment_settings(1)
	experiment.datasets = datasetObjs
	#experiments_pool.append(experiment)

	experiment = config_ND.load_experiment_settings(2)
	experiment.datasets = datasetObjs
	#experiments_pool.append(experiment)

	experiment = config_ND.load_experiment_settings(3)
	experiment.datasets = datasetObjs
	experiments_pool.append(experiment)

	#Running experiments: parallel or not
	if parallel_execution:
		#Parallelizing the experiments (optional): one experiment per Core
		pool = Pool()
		processes_pool = []

		for experiment in experiments_pool:
			processes_pool.append(pool.apipe(experiment.evaluator.evaluate, 
				repetitions, experiment.acronym, experiment.models, experiment.datasets, experiment.monitors))  
		
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
			avg_acc, avg_time, avg_cf, avg_mem, avg_f1 = experiment.evaluator.evaluate(
				repetitions, experiment.acronym, experiment.models, experiment.datasets, experiment.monitors) 
			
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