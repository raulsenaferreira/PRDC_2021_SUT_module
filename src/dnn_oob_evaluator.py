from src.utils import metrics
from time import perf_counter as timer
import numpy as np


def evaluate(repetitions, experiment_acronym, modelsObj, datasetObjs, monitorsObj):
	avg_acc = [experiment_acronym] #accuracy
	avg_cf = [experiment_acronym] #confusion matrix
	avg_time = [experiment_acronym] #time
	avg_memory = [experiment_acronym] #memory
	avg_F1 = [experiment_acronym] #memory
	
	
	for dataset, model, monitor in zip(datasetObjs, modelsObj, monitorsObj):
		acc = []
		t = []
		cf = [[],[],[],[]]
		mem = []
		f1 = []
		datasets = []
		
		for i in range(repetitions):
			print("{} experiment {} of {} ...".format(dataset.dataset_name, i+1, repetitions))
			X_test, y_test = dataset.load_dataset(mode='test')

			ini = timer()
			arrPred, arrLabel, memory, arrFP, arrFN, arrTP, arrTN = model.exec.run(
				X_test, y_test, model, monitor)
			end = timer()

			acc.append(metrics.evaluate(arrLabel, arrPred))
			t.append(end-ini)
			mem.append(memory)
			f1.append(metrics.F1(arrLabel, arrPred))

			classToMonitor = str(monitor.classToMonitor)
			cf[0].append(arrFP[classToMonitor])
			cf[1].append(arrFN[classToMonitor])
			cf[2].append(arrTP[classToMonitor])
			cf[3].append(arrTN[classToMonitor])
		
		avg_acc.append(np.mean(acc))
		avg_time.append(np.mean(t))
		avg_memory.append(np.mean(mem))
		avg_F1.append(np.mean(f1))
		avg_cf.append([np.mean(cf[0]), np.mean(cf[1]), np.mean(cf[2]), np.mean(cf[3])])

		'''
		acc = []
		t = []
		cf = [[],[],[],[]]
		mem = []
		f1 = []

		for i in range(repetitions):
			print("GTRSB experiment {} of {} ...".format(i+1, repetitions))
			model_name = 'DNN_GTRSB.h5'
			monitor_name = "monitor_Box_GTRSB.p"

			ini = timer()
			arrPred, arrLabel, count, arrFP, arrFN, arrTP, arrTN = DNN_outOfBox_GTRSB_test.run(
				expObj.classToMonitor, expObj.layer_name, expObj.models_folder, expObj.monitors_folder, monitor_name, model_name, 
				expObj.isTestOneClass, sep)
			end = timer()

			acc.append(metrics.evaluate(arrLabel, arrPred))
			t.append(end-ini)
			mem.append(memory)
			f1.append(metrics.F1(arrLabel, arrPred))
			cf[0].append(arrFP[str(expObj.classToMonitor)])
			cf[1].append(arrFN[str(expObj.classToMonitor)])
			cf[2].append(arrTP[str(expObj.classToMonitor)])
			cf[3].append(arrTN[str(expObj.classToMonitor)])

		avg_acc.append(np.mean(acc))
		avg_time.append(np.mean(t))
		avg_memory.append(np.mean(mem))
		avg_F1.append(np.mean(f1))
		avg_cf.append([np.mean(cf[0]), np.mean(cf[1]), np.mean(cf[2]), np.mean(cf[3])])
		datasets.append('GTSRB')
		'''
	return avg_acc, avg_time, avg_cf, avg_memory, avg_F1