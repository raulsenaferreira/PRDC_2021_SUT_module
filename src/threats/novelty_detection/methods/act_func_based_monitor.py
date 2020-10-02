import os
import numpy as np
import pickle
from src.utils import util


def build_monitor(model, X, y, class_to_monitor, layer_index, verbose):
	arrWeights = []

	#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
	counter = 0
	loading_percentage = 0.1
	loaded = int(loading_percentage*len(y))

	for img, lab in zip(X, y):
		if verbose:
			counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log

		lab = np.where(lab)[0]
		img = np.asarray([img])
		yPred = np.argmax(model.predict(img))
		
		if yPred == lab and yPred==class_to_monitor:
			arrWeights.append(util.get_activ_func(model, img, layerIndex=layer_index)[0])

	return arrWeights


def build_monitor_without_classify(model, X, y, class_to_monitor, layer_index, verbose):
	arrWeights = []

	#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
	counter = 0
	loading_percentage = 0.1
	loaded = int(loading_percentage*len(y))

	for img, lab in zip(X, y):
		if verbose:
			counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log

		lab = np.where(lab)[0]
		img = np.asarray([img])
		
		if lab == class_to_monitor:
			arrWeights.append(util.get_activ_func(model, img, layerIndex=layer_index)[0])

	return arrWeights


def run(monitor, model, X, y, save, params):
	arrWeights = []
	trained_monitor = None
	class_to_monitor = monitor.class_to_monitor
	layer_index = monitor.layer_index

	if params['use_alternative_monitor']:
		# building monitor with labels
		arrWeights = build_monitor_without_classify(model, X, y, class_to_monitor, layer_index, params['verbose'])
		monitor.filename += '_2'
	else:
		# building monitor with right predictions
		arrWeights = build_monitor(model, X, y, class_to_monitor, layer_index, params['verbose'])

	trained_monitor = monitor.method(arrWeights, monitor, save)

	file_path = os.path.join(monitor.monitors_folder, monitor.filename)

	if save:
		print("Saving monitor in", file_path)
		os.makedirs(monitor.monitors_folder, exist_ok=True)
		pickle.dump(trained_monitor, open( file_path, "wb" ))
	else:
		print("Monitor will not be saved")

	return trained_monitor