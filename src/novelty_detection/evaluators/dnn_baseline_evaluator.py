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
	csvs_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'csv'+sep+experiment.sub_field+sep+experiment.name+sep
	img_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'img'+sep+experiment.sub_field+sep+experiment.name+sep

	metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(experiment.name, arr_readouts, img_folder_path+'all_images.pdf')


def run_evaluation(experiment, repetitions, save_experiments):
	monitor_name = 'No monitor'
	arr_acc = [] #accuracy
	arr_t = [] #time
	arr_mem = [] #memory
	arr_f1 = [] #F1
	arr_mcc = [] #MCC
	arr_precision = []
	arr_recall = []

	dataset = experiment.dataset

	for i in range(repetitions):
		lbl_ypred_ood = [[], []] #ground truth and predicted

		print("Evaluating {}: {} of {} repetitions...\n".format(experiment.name, i+1, repetitions))
		
		ini = timer()
		arrPred, arrLabel, memory = experiment.tester.run(
			dataset.X, dataset.y, experiment, dataset.dataset_name)
		end = timer()

		arr_acc.append(metrics.evaluate(arrLabel, arrPred, 'accuracy'))
		arr_t.append(end-ini)
		arr_mem.append(memory)
		arr_f1.append(metrics.evaluate(arrLabel, arrPred, 'F1'))
		arr_mcc.append(metrics.evaluate(arrLabel, arrPred, 'MCC'))
		arr_precision.append(metrics.evaluate(arrLabel, arrPred, 'precision'))
		arr_recall.append(metrics.evaluate(arrLabel, arrPred, 'recall'))
		
	'''
	print('Accuracy', np.mean(arr_acc)) 
	print('Process time', np.mean(arr_t)) 
	print('Memory', np.mean(arr_mem))
	print('F1', np.mean(arr_f1))
	print('MCC', np.mean(arr_mcc))
	print('precision', np.mean(arr_precision))
	print('recall', np.mean(arr_recall))
	'''
	if save_experiments:
		neptune.create_experiment('hyper_parameter/{}'.format("baseline"))
		neptune.log_metric('Accuracy', np.mean(arr_acc)) 
		neptune.log_metric('Process time', np.mean(arr_t)) 
		neptune.log_metric('Memory', np.mean(arr_mem))
		neptune.log_metric('F1', np.mean(arr_f1))
		neptune.log_metric('MCC', np.mean(arr_mcc))
		neptune.log_metric('precision', np.mean(arr_precision))
		neptune.log_metric('recall', np.mean(arr_recall))

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
			processes_pool.append(pool.apipe(run_evaluation, experiment, repetitions, save_experiments))

		for process in processes_pool:
			success = process.get(timeout=timeout)
			#arr_readouts.append(readout)
	else:
		print("\nserial execution")
		success = run_evaluation(experiment, repetitions, save_experiments)

	#print('len(arr_readouts)', len(arr_readouts))
	#if save:
	#save_results(experiment, arr_readouts, plot=False)	
	return success
	#return arr_readouts


if __name__ == "__main__":
	pass