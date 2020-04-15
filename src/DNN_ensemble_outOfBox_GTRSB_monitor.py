import numpy as np
import pickle
from sklearn.cluster import KMeans
from src.utils import util
from src.utils import abstraction_box
from keras.models import load_model


def run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, validation_size, K, sep, script_path):
    models = []
    arrWeights = {}
    trainPath = str(script_path)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
    #loading German traffic sign dataset
    _, X_valid, _, Y_valid = util.load_GTRSB_dataset(trainPath, validation_size)

    #comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(Y_valid))

    # adding CNN trained with pre-processed images and preparing to store their weights
    for i in range(num_cnn):
        models.append(load_model(models_folder+model_ensemble_prefix+str(i)+'.h5'))
        arrWeights.update({i: []})

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
        pickle.dump(boxes, open(monitors_ensemble_folder+monitor_ensemble_prefix+str(i)+".p", "wb"))
    return True