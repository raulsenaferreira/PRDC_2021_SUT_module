from src.GTRSB_experiments import DNN_outOfBox_dimReduc_test
from src.GTRSB_experiments import DNN_ensemble_outOfBox_GTRSB_test
from src.GTRSB_experiments import DNN_outOfBox_GTRSB_test
#from src.MNIST_experiments import SCGAN_MNIST_test
from src.MNIST_experiments import dnn_oob_tester
from src.utils import abstraction_box
from src.utils import util
from src.utils import metrics
from src import dnn_oob_evaluator
from src import cnn_nl_oob_experiments
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from src.Classes.builder import ModelBuilder
from src.Classes.dataset import Dataset
from src.Classes.monitor import Monitor


# ML is incorrect but monitor does not trust on it = TP
# ML is correct but monitor does not trust on it = FP
# ML is incorrect and monitor trust on it = FN
# ML is correct and monitor trust on it = TN
if __name__ == "__main__":
	parallel_execution = True
	experiments_pool = []
	sep = util.get_separator()
	repetitions = 1
	experiment_acronym = 'CNN+OOB'
	dataset_names = ['MNIST', 'GTSRB']
	
	#models
	dnn_mnist = ModelBuilder()
	dnn_mnist.model_name = 'DNN_MNIST.h5'
	dnn_mnist.exec = dnn_oob_tester
	dnn_gtsrb = ModelBuilder()
	dnn_gtsrb.model_name = 'DNN_GTRSB.h5'
	dnn_gtsrb.exec = dnn_oob_tester

	modelsObj = [dnn_mnist, dnn_gtsrb]
	
	#datasets
	mnistObj = Dataset(dataset_names[0]) 
	gtsrbObj = Dataset(dataset_names[1]) 
	datasetObjs = [mnistObj, gtsrbObj]
	
	#monitors
	monitorObjMNIST = Monitor("monitor_Box_MNIST.p", 7, 'dense_1', -2)
	monitorObjMNIST.method = abstraction_box.find_point
	monitorObjGTSRB = Monitor("monitor_Box_GTRSB.p", 7, 'dense_1', -2)
	monitorObjGTSRB.method = abstraction_box.find_point
	monitorsObj = [monitorObjMNIST, monitorObjGTSRB]

	# specific variables for this results
	img_name = 'all_images.pdf'
	acc_file_name = 'accuracies.csv'
	cf_file_name = 'positive_negative_rates.csv'
	time_file_name = 'time.csv'
	mem_file_name = 'memory.csv'
	f1_file_name = 'f1.csv'
	csvs_folder_path = 'results'+sep+'csv'+sep
	img_folder_path = 'results'+sep+'img'+sep
	accuracies = []
	confusion_matrices = []
	proc_times = []
	memories = []
	f1s = []

	if parallel_execution:
		#Parallelizing the experiments (optional): one experiment per Core
		pool = Pool()
		#cnn_oob_experiments.run(repetitions, experiment_acronym, modelsObj, datasetObjs, monitorsObj)
		#Experiment 1: CNN with outside-of-box monitor
		experiments_pool.append(pool.apipe(dnn_oob_evaluator.evaluate, 
			repetitions, experiment_acronym, modelsObj, datasetObjs, monitorsObj))  
		'''
		#Experiment 2: CNN with outside-of-box monitor with non-linear dimensionality reduction
		experiments_pool.append(pool.apply_async(cnn_nl_oob_experiments.run, [repetitions, classToMonitor, 
			layer_index, layer_name, models_folder, monitors_folder, isTestOneClass, sep]))
		'''
		for experiment in experiments_pool:
			avg_acc, avg_time, avg_cf, avg_mem, avg_f1 = experiment.get(timeout=1200)

			accuracies.append(avg_acc)
			confusion_matrices.append(avg_cf)
			proc_times.append(avg_time)
			memories.append(avg_mem)
			f1s.append(avg_f1)		
	else:
		#Serial version for the experiments
		#Experiment 1: CNN with outside-of-box monitor
		experiments_pool.append(cnn_oob_experiments.run(repetitions, classToMonitor, 
			layer_name, models_folder, monitors_folder, isTestOneClass, sep))  
		
		#Experiment 2: CNN with outside-of-box monitor with non-linear dimensionality reduction
		experiments_pool.append(cnn_nl_oob_experiments.run(repetitions, classToMonitor, 
			layer_index, layer_name, models_folder, monitors_folder, isTestOneClass, sep))

		for experiment in experiments_pool:
			avg_acc, avg_time, avg_cf, avg_mem, avg_f1 = experiment
			accuracies.append(avg_acc)
			confusion_matrices.append(avg_time)
			proc_times.append(avg_cf)
			memories.append(avg_mem)
			f1s.append(avg_f1)

		#Experiment 2: ensemble of CNNs with outside-of-box monitor
		#model_ensemble_prefix = 'DNN_ensemble_GTRSB_'
		#num_cnn = 5
		#monitors_ensemble_folder = "src"+sep+"bin"+sep+"monitors"+sep+"outOfBox_ensembleDNN"+sep
		#monitor_ensemble_prefix = "monitor_Box_DNN_"
		#arrPred, count, arrFP, arrFN, arrTP, arrTN = DNN_ensemble_outOfBox_GTRSB_test.run(classToMonitor, layer_index, 
		#	models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, 
		#	sep, script_path)

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
	#print('confusion_matrices', confusion_matrices)
	metrics.plot_pos_neg_rate_stacked_bars(confusion_matrices, dataset_names, img_folder_path+img_name)