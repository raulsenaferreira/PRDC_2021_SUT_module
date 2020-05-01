import numpy as np
import pickle
from src.utils import util
from keras.models import load_model


def run(monitor, model_obj):
	K = 3
	arrWeights = []
	#loading dataset
	_, _, x_valid, y_valid = monitor.dataset.load_dataset(mode='train')

	#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
	counter = 0
	loading_percentage = 0.1
	loaded = int(loading_percentage*len(y_valid))

	model = load_model(model_obj.models_folder+model_obj.model_name)

	#building monitor with validation dataset
	for img, lab in zip(x_valid, y_valid):
		lab = np.where(lab)[0]
		counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
		img = np.asarray([img])
		yPred = np.argmax(model.predict(img))
		
		if util.is_monitored_prediction(yPred, lab, monitor.classToMonitor):
			arrWeights.append(util.get_activ_func(model, img, layerIndex=monitor.layer_index)[0])
	
	trained_monitor = monitor.method(arrWeights, K, monitor.classToMonitor)
	file = monitor.monitors_folder+monitor.monitor_name
	print("Saving monitor in", file)
	pickle.dump(trained_monitor, open( file, "wb" ))

	return True