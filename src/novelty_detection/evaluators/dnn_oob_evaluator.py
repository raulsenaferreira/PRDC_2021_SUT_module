from src.utils import metrics
from time import perf_counter as timer
import numpy as np
from src.Classes.readout import Readout 


def evaluate(repetitions, experiment, percentage_of_data=1):
	arr_readouts = []
	
	dataset = experiment.dataset
	
	for monitor in experiment.monitors:
		arr_acc = [] #accuracy
		arr_cf = [[],[],[],[]] #confusion matrix
		arr_t = [] #time
		arr_mem = [] #memory
		arr_f1 = [] #F1
		
		for i in range(repetitions):
			print("{} with '{}' monitor: {} of {} ...".format(experiment.name, monitor.monitor_name, i+1, repetitions))
			X_test, y_test = dataset.load_dataset(mode='test')
			
			# for one that wants speeding up tests using part of data
			X_limit = int(len(X_test)*percentage_of_data)
			y_limit = int(len(y_test)*percentage_of_data)
			X_test, y_test = X_test[: X_limit], y_test[: y_limit]

			ini = timer()
			arrPred, arrLabel, memory, arrFP, arrFN, arrTP, arrTN = experiment.tester.run(
				X_test, y_test, experiment.model, monitor, dataset.dataset_name)
			end = timer()

			arr_acc.append(metrics.evaluate(arrLabel, arrPred))
			arr_t.append(end-ini)
			arr_mem.append(memory)
			arr_f1.append(metrics.F1(arrLabel, arrPred))

			arr_cf[0].append(arrFP[monitor.class_to_monitor])
			arr_cf[1].append(arrFN[monitor.class_to_monitor])
			arr_cf[2].append(arrTP[monitor.class_to_monitor])
			arr_cf[3].append(arrTN[monitor.class_to_monitor])
		
		# storing results
		readout = Readout()
		readout.name = monitor.monitor_name
		readout.avg_acc = np.mean(arr_acc) 
		readout.avg_time = np.mean(arr_t) 
		readout.avg_memory = np.mean(arr_mem)
		readout.avg_F1 = np.mean(arr_f1)
		readout.avg_cf = [int(np.mean(arr_cf[0])), int(np.mean(arr_cf[1])),
		 int(np.mean(arr_cf[2])), int(np.mean(arr_cf[3]))]

		arr_readouts.append(readout)

	return arr_readouts