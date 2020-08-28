import os
import neptune
from src.Classes.readout import Readout
from src.utils import metrics
from src.utils import util

sep = util.get_separator()

#saving experiments in the cloud (optional)
project = neptune.init('raulsenaferreira/PhD')

# Get list of experiments
experiments = project.get_experiments(id=['PHD-24', 'PHD-25', 'PHD-26'])
names = ['oob', 'oob_isomap', 'oob_pca']
dataset_name = 'GTSRB'

arr_readouts = []
classes_to_monitor = 43
img_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'img'+sep 

for experiment, name in zip(experiments, names):
	avg_cf = {}

	logs = experiment.get_logs()
	#print(logs['True Positive - Class 0'].y) 

	# storing results
	readout = Readout()
	readout.name = name
	
	readout.avg_acc = logs['Accuracy'].y
	readout.avg_time = logs['Process time'].y
	readout.avg_memory = logs['Memory'].y
	readout.avg_F1 = logs['F1'].y

	for class_to_monitor in range(classes_to_monitor):
		fp = 'False Positive - Class {}'.format(class_to_monitor)
		fn = 'False Negative - Class {}'.format(class_to_monitor)
		tp = 'True Positive - Class {}'.format(class_to_monitor)
		tn = 'True Negative - Class {}'.format(class_to_monitor)

		avg_cf.update({class_to_monitor: [int(float(logs[fp].y)), int(float(logs[fn].y)), int(float(logs[tp].y)), int(float(logs[tn].y))]})
	readout.avg_cf = avg_cf

	arr_readouts.append(readout)

fig_name = img_folder_path+'all_methods_class_'+dataset_name+'.pdf'
os.makedirs(img_folder_path, exist_ok=True)
metrics.plot_pos_neg_rate_stacked_bars_total(dataset_name, arr_readouts, classes_to_monitor, fig_name)