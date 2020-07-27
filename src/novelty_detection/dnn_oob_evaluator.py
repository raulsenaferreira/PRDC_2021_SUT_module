from src.utils import metrics
from time import perf_counter as timer
import numpy as np


def evaluate(repetitions, experiment):
	avg_acc = [experiment.name] #accuracy
	avg_cf = [experiment.name] #confusion matrix
	avg_time = [experiment.name] #time
	avg_memory = [experiment.name] #memory
	avg_F1 = [experiment.name] #memory
	
	dataset = experiment.dataset
	model = experiment.model
	
	for monitor in experiment.monitors:
		acc = []
		t = []
		cf = [[],[],[],[]]
		mem = []
		f1 = []

		classToMonitor = monitor.classToMonitor
		
		for i in range(repetitions):
			print("{}: {} of {} ...".format(experiment.name, i+1, repetitions))
			X_test, y_test = dataset.load_dataset(mode='test')

			ini = timer()
			arrPred, arrLabel, memory, arrFP, arrFN, arrTP, arrTN = experiment.tester.run(
				X_test, y_test, model.binary, monitor)
			end = timer()

			acc.append(metrics.evaluate(arrLabel, arrPred))
			t.append(end-ini)
			mem.append(memory)
			f1.append(metrics.F1(arrLabel, arrPred))

			cf[0].append(arrFP[classToMonitor])
			cf[1].append(arrFN[classToMonitor])
			cf[2].append(arrTP[classToMonitor])
			cf[3].append(arrTN[classToMonitor])
		
		avg_acc.append(np.mean(acc))
		avg_time.append(np.mean(t))
		avg_memory.append(np.mean(mem))
		avg_F1.append(np.mean(f1))
		avg_cf.append([np.mean(cf[0]), np.mean(cf[1]), np.mean(cf[2]), np.mean(cf[3])])

	return avg_acc, avg_time, avg_cf, avg_memory, avg_F1