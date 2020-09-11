import os
from src.utils import metrics
from time import perf_counter as timer
import numpy as np
from src.Classes.readout import Readout
from src.utils import metrics
from src.novelty_detection import config as config_ND
from src.utils import util
from pathos.multiprocessing import ProcessingPool as Pool
import neptune



sep = util.get_separator()

def save_results(experiment, arr_readouts, plot=False):
	print("saving experiments", experiment.name)
	filenames = config_ND.load_file_names()
	csvs_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'csv'+sep+experiment.experiment_type+sep+experiment.name+sep
	img_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'img'+sep+experiment.experiment_type+sep+experiment.name+sep

	metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(experiment.name, arr_readouts, img_folder_path+'all_images.pdf')


def run_evaluation(monitor, experiment, repetitions, save_experiments):
	arr_acc = [] #accuracy
	arr_cf_ID = {} #confusion matrix by class for ID data
	arr_cf_OOD = {} #confusion matrix by class for OOD data
	arr_t = [] #time
	arr_mem = [] #memory
	arr_f1 = [] #F1

	for class_to_monitor in range(experiment.classes_to_monitor):
		arr_cf_ID.update({class_to_monitor: [[],[],[],[]]})
		arr_cf_OOD.update({class_to_monitor: [[],[]]})

	dataset = experiment.dataset

	for i in range(repetitions):
		print("Evaluating {} with {} monitor: {} of {} repetitions...\n".format(experiment.name, monitor.monitor_name, i+1, repetitions))
		
		ini = timer()
		arrPred, arrLabel, memory, arrFP_ID, arrFN_ID, arrTP_ID, arrTN_ID, arrFN_OOD, arrTP_OOD = experiment.tester.run(
			dataset.X, dataset.y, experiment, monitor, dataset.dataset_name)
		end = timer()

		arr_acc.append(metrics.evaluate(arrLabel, arrPred, 'accuracy'))
		arr_t.append(end-ini)
		arr_mem.append(memory)
		arr_f1.append(metrics.evaluate(arrLabel, arrPred, 'F1'))
	
		for class_to_monitor in range(experiment.classes_to_monitor):
			# ID
			arr_cf_ID[class_to_monitor][0].append(arrFP_ID[class_to_monitor])
			arr_cf_ID[class_to_monitor][1].append(arrFN_ID[class_to_monitor])
			arr_cf_ID[class_to_monitor][2].append(arrTP_ID[class_to_monitor])
			arr_cf_ID[class_to_monitor][3].append(arrTN_ID[class_to_monitor])
			# OOD
			arr_cf_OOD[class_to_monitor][0].append(len(arrFN_OOD[class_to_monitor]))
			arr_cf_OOD[class_to_monitor][1].append(len(arrTP_OOD[class_to_monitor]))	

	if save_experiments:
		neptune.create_experiment('hyper_parameter/{}'.format(monitor.monitor_name))
		neptune.log_metric('Accuracy', np.mean(arr_acc)) 
		neptune.log_metric('Process time', np.mean(arr_t)) 
		neptune.log_metric('Memory', np.mean(arr_mem))
		neptune.log_metric('F1', np.mean(arr_f1))

		for monitored_class in range(experiment.classes_to_monitor):
			# ID + OOD
			fn_OOD = int(np.mean(arr_cf_OOD[monitored_class][0]))
			tp_OOD = int(np.mean(arr_cf_OOD[monitored_class][1]))

			neptune.log_metric('False Positive - Class {}'.format(monitored_class), int(np.mean(arr_cf_ID[monitored_class][0])))
			neptune.log_metric('False Negative - Class {}'.format(monitored_class), int(np.mean(arr_cf_ID[monitored_class][1])) + fn_OOD)
			neptune.log_metric('True Positive - Class {}'.format(monitored_class), int(np.mean(arr_cf_ID[monitored_class][2])) + tp_OOD)
			neptune.log_metric('True Negative - Class {}'.format(monitored_class), int(np.mean(arr_cf_ID[monitored_class][3])))

	return True


def evaluate(repetitions, experiment, parallel_execution, save_experiments):
	cores = 6
	arr_readouts = []
	processes_pool = []
	success = False

	if  parallel_execution:
		pool = Pool(cores)
		timeout = 1000 #* len(experiment.monitors)
		print("\nParallel execution with {} cores. Max {} seconds to run each experiment:".format(cores, timeout))

		for monitor in experiment.monitors:
			processes_pool.append(pool.apipe(run_evaluation, monitor, experiment, repetitions, save_experiments))

		for process in processes_pool:
			success = process.get(timeout=timeout)
			#arr_readouts.append(readout)
	else:
		print("\nserial execution")
		for monitor in experiment.monitors:
			success = run_evaluation(monitor, experiment, repetitions, save_experiments)
			#arr_readouts.append(readout)

	#print('len(arr_readouts)', len(arr_readouts))
	#if save:
	#save_results(experiment, arr_readouts, plot=False)	
	return success
	#return arr_readouts


if __name__ == "__main__":
	pass