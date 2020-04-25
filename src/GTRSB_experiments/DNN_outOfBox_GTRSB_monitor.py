import numpy as np
import pickle
from sklearn.cluster import KMeans
from src.utils import util
from src.utils import abstraction_box
from keras.models import load_model


def run(classToMonitor, monitors_folder, monitor_name, models_folder, model_name, layer_name, validation_size, K, sep):
	arrWeights = []
	

	#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
	counter = 0
	loading_percentage = 0.1
	loaded = int(loading_percentage*len(Y_valid))

	model = load_model(models_folder+model_name)

	#building monitor with validation dataset
	for img, lab in zip(X_valid, Y_valid):
		lab = np.where(lab)[0]
		counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
		img = np.asarray([img])
		yPred = np.argmax(model.predict(img))
		
		if yPred == lab and yPred==classToMonitor:
			arrWeights.append(util.get_activ_func(model, img, layerName=layer_name)[0])

	clusters = KMeans(n_clusters=K, random_state=0).fit_predict(arrWeights)
	print("making boxes...")
	boxes = abstraction_box.make_abstraction(arrWeights, clusters, classToMonitor)
	print("Saving boxes in a file...")
	pickle.dump(boxes, open( monitors_folder+monitor_name, "wb" ))

	return True