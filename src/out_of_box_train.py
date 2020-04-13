import sys
import os
from pathlib import Path
import numpy as np
import pickle
from sklearn.cluster import KMeans
from utils import util
from utils import abstraction_box
from keras.models import load_model


is_windows = sys.platform.startswith('win')
sep = '\\'

if is_windows == False:
    sep = '/'

classToMonitor = 7
layer_name = 'dense_1'
K = 3
validation_size = 0.3
arrWeights = []
models_folder = "bin"+sep+"models"+sep
monitors_folder = "bin"+sep+"monitors"+sep
script_path = os.getcwd()
trainPath = str(Path(script_path).parent)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
#loading German traffic sign dataset
X_train,X_valid,Y_train,Y_valid = util.load_GTRSB_dataset(trainPath, validation_size)

#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
counter = 0
loading_percentage = 0.1
loaded = int(loading_percentage*len(Y_valid))

model = load_model(models_folder+'CNN_GTRSB.h5')

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
pickle.dump(boxes, open( monitors_folder+"monitor_Box_GTRSB.p", "wb" ))