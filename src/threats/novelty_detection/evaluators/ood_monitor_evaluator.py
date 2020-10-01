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
	arr_acc = [] #accuracy
	arr_t = [] #time
	arr_mem = [] #memory
	arr_f1 = [] #F1
	arr_cf_ID = {} #confusion matrix by class for ID data
	arr_cf_OOD = {} #confusion matrix by class for OOD data

	for class_to_monitor in range(experiment.classes_to_monitor_ID):
		arr_cf_ID.update({class_to_monitor: [[],[],[],[]]})

	for class_OOD in range(experiment.classes_to_monitor_ID, experiment.classes_to_monitor_OOD + experiment.classes_to_monitor_ID):
		arr_cf_OOD.update({class_OOD: [[],[],[],[]]})

	dataset = experiment.dataset

	# only for boxes-based monitors enlarged with a constant 'tau' 
	try: 
		monitor.monitor_name += '_tau_'+str(monitor.tau)
	except:
		pass

	for i in range(repetitions):
		print("Evaluating {} with {} monitor: {} of {} repetitions...\n".format(experiment.name, monitor.monitor_name, i+1, repetitions))
		
		ini = timer()
		arrPred, arrLabel, readout = experiment.tester.run(dataset, experiment, monitor)
		end = timer()

		arr_acc.append(metrics.evaluate(arrLabel, arrPred, 'accuracy'))
		arr_f1.append(metrics.evaluate(arrLabel, arrPred, 'F1'))
		arr_t.append(end-ini)
		arr_mem.append(readout.memory)
		arr_pos_neg_ID_pred = readout.arr_pos_neg_ID_pred
		arr_pos_neg_ID_true = readout.arr_pos_neg_ID_true
		arr_pos_neg_OOD_true = readout.arr_pos_neg_OOD_true
		arr_pos_neg_OOD_pred = readout.arr_pos_neg_OOD_pred
		
		for class_to_monitor in range(experiment.classes_to_monitor_ID):
			# ID
			arr_cf_ID[class_to_monitor][0].append(len(readout.arr_false_positive_ID[class_to_monitor]))
			arr_cf_ID[class_to_monitor][1].append(len(readout.arr_false_negative_ID[class_to_monitor]))
			arr_cf_ID[class_to_monitor][2].append(len(readout.arr_true_positive_ID[class_to_monitor]))
			arr_cf_ID[class_to_monitor][3].append(len(readout.arr_true_negative_ID[class_to_monitor]))

		
		for class_OOD in range(experiment.classes_to_monitor_ID, experiment.classes_to_monitor_OOD + experiment.classes_to_monitor_ID):
			# OOD
			arr_cf_OOD[class_OOD][0].append(len(readout.arr_false_positive_OOD[class_OOD]))
			arr_cf_OOD[class_OOD][1].append(len(readout.arr_false_negative_OOD[class_OOD]))
			arr_cf_OOD[class_OOD][2].append(len(readout.arr_true_positive_OOD[class_OOD]))
			arr_cf_OOD[class_OOD][3].append(len(readout.arr_true_negative_OOD[class_OOD]))
	'''
	#### for test purposes
	import random 
	readout2 = copy.deepcopy(readout)
	random.shuffle(readout2.arr_pos_neg_ID_pred)
	random.shuffle(readout2.arr_pos_neg_OOD_pred)
	readout.title, readout2.title = 'Method 1', 'Method 2'
	readout.id_dataset, readout2.id_dataset = 'GTSRB', 'GTSRB'
	readout.ood_dataset, readout2.ood_dataset = 'BTSC', 'BTSC'
	metrics.plot_ROC_curve_ID_OOD([readout, readout2], 'fp_tp')
	#### 
	'''
	if save_experiments:
		tag1 = monitor.monitor_name
		tag2 = 'ID = {}'.format(dataset.dataset_ID_name)
		tag3 = 'OOD = {}'.format(dataset.dataset_OOD_name)

		neptune.create_experiment(name='{}'.format(experiment.name),
										tags=[tag1, tag2, tag3],
										params=experiment.PARAMS)

		util.save_metrics_neptune(neptune,
			['ML_accuracy', 'Process_time', 'Memory', 'ML_F1'],
			[np.mean(arr_acc), np.mean(arr_t), np.mean(arr_mem), np.mean(arr_f1)]
			)
		'''
		neptune.log_metric('ML_accuracy', np.mean(arr_acc)) 
		neptune.log_metric('Process_time', np.mean(arr_t)) 
		neptune.log_metric('Memory', np.mean(arr_mem))
		neptune.log_metric('ML_F1', np.mean(arr_f1))

		# ID
		
		tmp_path = os.path.join('results', 'temp')
		os.makedirs(tmp_path, exist_ok=True)

		tmp_path = os.path.join(tmp_path, 'Pos_Neg_Classified_ID.npy')
		np.save(tmp_path, arr_pos_neg_ID_pred) #np.load('Pos_Neg_Classified_ID.npy')
		neptune.log_artifact(tmp_path)
		os.remove(tmp_path)

		tmp_path = os.path.join('results', 'temp', 'Pos_Neg_Labels_ID.npy')
		np.save(tmp_path, arr_pos_neg_ID_true) #np.load('Pos_Neg_Labels_ID.npy')
		neptune.log_artifact(tmp_path)
		os.remove(tmp_path)
		'''
		util.save_artifact_neptune(neptune, 'Pos_Neg_Classified_ID.npy', arr_pos_neg_ID_pred)
		util.save_artifact_neptune(neptune, 'Pos_Neg_Labels_ID.npy', arr_pos_neg_ID_true)

		for monitored_class in range(experiment.classes_to_monitor_ID):
			aux = '_ID_{}'.format(monitored_class)

			util.save_metrics_neptune(neptune,
				['False_Positive'+aux, 'False_Negative'+aux, 'True_Positive'+aux, 'True_Negative'+aux],
				[int(np.mean(arr_cf_ID[monitored_class][0])), int(np.mean(arr_cf_ID[monitored_class][1])), 
				int(np.mean(arr_cf_ID[monitored_class][2])), int(np.mean(arr_cf_ID[monitored_class][3]))]
			)
			
			'''
			neptune.log_metric('False_Positive_ID_{}'.format(monitored_class), int(np.mean(arr_cf_ID[monitored_class][0])))
			neptune.log_metric('False_Negative_ID_{}'.format(monitored_class), int(np.mean(arr_cf_ID[monitored_class][1])))
			neptune.log_metric('True_Positive_ID_{}'.format(monitored_class), int(np.mean(arr_cf_ID[monitored_class][2])))
			neptune.log_metric('True_Negative_ID_{}'.format(monitored_class), int(np.mean(arr_cf_ID[monitored_class][3])))
			'''
		# OOD
		
		'''
		tmp_path = os.path.join('results', 'temp', 'Pos_Neg_Classified_OOD.npy')
		np.save(tmp_path, arr_pos_neg_OOD_pred) #np.load('Pos_Neg_Classified_OOD.npy')
		neptune.log_artifact(tmp_path)
		os.remove(tmp_path)

		tmp_path = os.path.join('results', 'temp', 'Pos_Neg_Labels_OOD.npy')
		np.save(tmp_path, arr_pos_neg_OOD_true) #np.load('Pos_Neg_Labels_OOD.npy')
		neptune.log_artifact(tmp_path)
		os.remove(tmp_path)
		'''
		util.save_artifact_neptune(neptune, 'Pos_Neg_Classified_OOD.npy', arr_pos_neg_OOD_pred)
		util.save_artifact_neptune(neptune, 'Pos_Neg_Labels_OOD.npy', arr_pos_neg_OOD_true)

		for class_OOD in range(experiment.classes_to_monitor_ID, experiment.classes_to_monitor_OOD + experiment.classes_to_monitor_ID):
			aux = '_OOD_{}'.format(class_OOD)

			util.save_metrics_neptune(neptune,
				['False_Positive'+aux, 'False_Negative'+aux, 'True_Positive'+aux, 'True_Negative'+aux],
				[int(np.mean(arr_cf_OOD[class_OOD][0])), int(np.mean(arr_cf_OOD[class_OOD][1])), 
				int(np.mean(arr_cf_OOD[class_OOD][2])), int(np.mean(arr_cf_OOD[class_OOD][3]))]
			)
			'''
			neptune.log_metric('False_Positive_OOD_{}'.format(class_OOD), int(np.mean(arr_cf_OOD[class_OOD][0])))
			neptune.log_metric('False_Negative_OOD_{}'.format(class_OOD), int(np.mean(arr_cf_OOD[class_OOD][1])))
			neptune.log_metric('True_Positive_OOD_{}'.format(class_OOD), int(np.mean(arr_cf_OOD[class_OOD][2])))
			neptune.log_metric('True_Negative_OOD_{}'.format(class_OOD), int(np.mean(arr_cf_OOD[class_OOD][3])))
			'''
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