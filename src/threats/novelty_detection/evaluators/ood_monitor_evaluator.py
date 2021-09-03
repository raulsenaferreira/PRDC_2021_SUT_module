import io
import os
import copy
from src.utils import metrics
from time import perf_counter as timer
import numpy as np
from src.Classes.readout import Readout
from src.utils import metrics
from src.threats.novelty_detection import config as config_ND
from src.utils import util
from pathos.multiprocessing import ProcessingPool as Pool
import neptune


def run_evaluation(monitor, experiment, repetitions, save_experiments):
	arr_pos_neg_ID_pred, arr_pos_neg_ID_true = None, None
	arr_pos_neg_OOD_true, arr_pos_neg_OOD_pred = None, None
	arr_t = [] #time
	arr_mem = [] #memory
	
	dataset = experiment.dataset

	print("Evaluating {} for {} with {} ...\n".format(experiment.name, dataset.dataset_ID_name, dataset.modification))
	
	ini = timer()
	readout = experiment.tester.run(dataset, experiment, monitor)
	end = timer()

	# general readouts
	readout.total_time = end-ini

	# printing some results
	print('Results:\n Detected {} OOD instances of {}'.format(sum(readout.arr_detection_SM), sum(readout.arr_detection_true)))
	
	if save_experiments:
		tag1 = monitor.monitor_name
		tag2 = 'ID = {}'.format(dataset.dataset_ID_name)
		tag3 = 'OOD = {}'.format(dataset.dataset_OOD_name)
		#tag4 = 'tau = {}'.format(monitor.tau)

		neptune.create_experiment(name='{}'.format(experiment.name),
										tags=[tag1, tag2, tag3],
										params=experiment.PARAMS)

		util.save_metrics_neptune(
			neptune,
			['ml_time', 'sm_time', 'total_time', 'total_memory'],
			[np.sum(readout.ML_time), np.sum(readout.SM_time), readout.total_time, readout.total_memory]
		)
		
		# ML readouts
		util.save_artifact_neptune(neptune, 'arr_classification_pred.npy', readout.arr_classification_pred)
		util.save_artifact_neptune(neptune, 'arr_classification_true.npy', readout.arr_classification_true)
		# SM readouts
		util.save_artifact_neptune(neptune, 'arr_detection_SM.npy', readout.arr_detection_SM)
		util.save_artifact_neptune(neptune, 'arr_detection_true.npy', readout.arr_detection_true)
		util.save_artifact_neptune(neptune, 'arr_reaction_SM.npy', readout.arr_reaction_SM)
		util.save_artifact_neptune(neptune, 'arr_reaction_true.npy', readout.arr_reaction_true)
			
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
	else:
		print("\nserial execution")
		for monitor in experiment.monitors:
			success = run_evaluation(monitor, experiment, repetitions, save_experiments)
	
	return success


if __name__ == "__main__":
	pass