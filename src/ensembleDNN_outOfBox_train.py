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
layer_index = 8
K = 3
validation_size = 0.3
arrWeights = []
models = []
arrWeights = {}
num_cnn = 4 #number of CNN made with data pre-processed 

models_folder = "bin"+sep+"models"+sep
monitors_folder = "bin"+sep+"monitors"+sep
script_path = os.getcwd()
trainPath = str(Path(script_path).parent)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
#loading German traffic sign dataset
_, X_valid, _, Y_valid = util.load_GTRSB_dataset(trainPath, validation_size)

#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
counter = 0
loading_percentage = 0.1
loaded = int(loading_percentage*len(Y_valid))

# adding CNN trained with pre-processed images and preparing to store their weights
for i in range(num_cnn):
    models.append(load_model(models_folder+'CNN_ensemble_GTRSB_'+str(i)+'.h5'))
    arrWeights.update({i: []})

# adding CNN trained with the original images and preparing to store it's weights
models.append(load_model(models_folder+'CNN_GTRSB.h5'))
arrWeights.update({num_cnn: []})
num_cnn += 1

for img, lab in zip(X_valid, Y_valid):
    lab = np.where(lab)[0]
    counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
    img = np.asarray([img])
    #aplying ensemble
    y_all = np.vstack((
        models[0].predict(img)[0], models[1].predict(img)[0], models[2].predict(img)[0], 
        models[3].predict(img)[0], models[4].predict(img)[0]
        ))
    
    y_all = np.average(y_all, axis=0)
    yPred = np.argmax(y_all)
    
    if yPred == lab and yPred==classToMonitor:
        for i in range(num_cnn):
            arrWeights[i].append(util.get_activ_func(models[i], img, layerIndex=layer_index)[0])
            
print("making boxes and saving in a file...")
for i in range(num_cnn):
    clusters = KMeans(n_clusters=K, random_state=0).fit_predict(arrWeights[i])
    boxes = abstraction_box.make_abstraction(arrWeights[i], clusters, classToMonitor)
    pickle.dump(boxes, open(monitors_folder+"outOfBox_ensembleDNN"+sep+"monitor_Box_DNN_"+str(i)+".p", "wb"))