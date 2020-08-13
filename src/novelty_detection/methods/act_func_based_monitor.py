import os
import numpy as np
import pickle
from src.utils import util


def build_monitor(model, X, y, class_to_monitor, layer_index):
	arrWeights = []

	#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
	counter = 0
	loading_percentage = 0.1
	loaded = int(loading_percentage*len(y))

	for img, lab in zip(X, y):
		lab = np.where(lab)[0]
		counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
		img = np.asarray([img])
		yPred = np.argmax(model.predict(img))
		
		if util.is_monitored_prediction(yPred, lab, class_to_monitor):
			arrWeights.append(util.get_activ_func(model, img, layerIndex=layer_index)[0])

	return arrWeights

def run(monitor, model, X, y):
	class_to_monitor = monitor.class_to_monitor
	layer_index = monitor.layer_index

	#building monitor with training set
	arrWeights = build_monitor(model, X, y, class_to_monitor, layer_index)
	
	trained_monitor = monitor.method(arrWeights, monitor)
	
	file_path = monitor.monitors_folder+monitor.filename
	print("Saving monitor in", file_path)
	os.makedirs(monitor.monitors_folder, exist_ok=True)
	pickle.dump(trained_monitor, open( file_path, "wb" ))

	return trained_monitor