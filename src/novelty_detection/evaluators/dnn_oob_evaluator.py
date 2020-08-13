import os
from src.utils import metrics
from time import perf_counter as timer
import numpy as np
from src.Classes.readout import Readout
from src.utils import metrics
from src.novelty_detection import config as config_ND
from src.utils import util
from pathos.multiprocessing import ProcessingPool as Pool



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


def run_evaluation(monitor, experiment, repetitions):
	arr_acc = [] #accuracy
	arr_cf = {} #confusion matrix by class
	arr_t = [] #time
	arr_mem = [] #memory
	arr_f1 = [] #F1

	for class_to_monitor in range(experiment.classes_to_monitor):
		arr_cf.update({class_to_monitor: [[],[],[],[]]})

	dataset = experiment.dataset

	for i in range(repetitions):
		print("Evaluating {} with {} monitor: {} of {} ...\n".format(experiment.name, monitor.monitor_name, i+1, repetitions))
		
		ini = timer()
		arrPred, arrLabel, memory, arrFP, arrFN, arrTP, arrTN = experiment.tester.run(
			dataset.X, dataset.y, experiment, monitor, dataset.dataset_name)
		end = timer()

		arr_acc.append(metrics.evaluate(arrLabel, arrPred))
		arr_t.append(end-ini)
		arr_mem.append(memory)
		arr_f1.append(metrics.F1(arrLabel, arrPred))
	
		for class_to_monitor in range(experiment.classes_to_monitor):
			arr_cf[class_to_monitor][0].append(arrFP[class_to_monitor])
			arr_cf[class_to_monitor][1].append(arrFN[class_to_monitor])
			arr_cf[class_to_monitor][2].append(arrTP[class_to_monitor])
			arr_cf[class_to_monitor][3].append(arrTN[class_to_monitor])	

	# storing results
	readout = Readout()
	readout.name = monitor.monitor_name
	
	readout.avg_acc = np.mean(arr_acc) 
	readout.avg_time = np.mean(arr_t) 
	readout.avg_memory = np.mean(arr_mem)
	readout.avg_F1 = np.mean(arr_f1)

	avg_cf = {}
	for class_to_monitor in range(experiment.classes_to_monitor):
		avg_cf[class_to_monitor] = [
		 int(np.mean(arr_cf[class_to_monitor][0])),
		 int(np.mean(arr_cf[class_to_monitor][1])),
		 int(np.mean(arr_cf[class_to_monitor][2])),
		 int(np.mean(arr_cf[class_to_monitor][3]))
		]
	readout.avg_cf = avg_cf

	return readout


def evaluate(repetitions, experiment, parallel_execution):
	arr_readouts = []
	processes_pool = []

	if  parallel_execution:
		pool = Pool()
		timeout = 600 #* len(experiment.monitors)
		print("\nMax seconds to run each experiment:", timeout)

		for monitor in experiment.monitors:
			processes_pool.append(pool.apipe(run_evaluation, monitor, experiment, repetitions))

		for process in processes_pool:
			readout = process.get(timeout=timeout)
			arr_readouts.append(readout)
	else:
		for monitor in experiment.monitors:
			readout = run_evaluation(monitor, experiment, repetitions)
			arr_readouts.append(readout)

	print('len(arr_readouts)', len(arr_readouts))
	#if save:
	#save_results(experiment, arr_readouts, plot=False)	

	return arr_readouts


if __name__ == "__main__":
	pass