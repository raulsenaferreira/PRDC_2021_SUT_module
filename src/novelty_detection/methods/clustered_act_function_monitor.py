import os
import numpy as np
import pickle
from src.utils import util
from sklearn.neighbors import KNeighborsClassifier
import hdbscan
from sklearn.preprocessing import StandardScaler



def build_monitor(model, X, y, layer_index):
    arrWeights = []
    arrLabels = []

    #comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y))

    for img, lab in zip(X, y):
        lab = np.where(lab)[0]
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        
        if yPred == lab:
            arrWeights.append(util.get_activ_func(model, img, layerIndex=layer_index)[0])
            arrLabels.append(lab)

    return arrWeights, arrLabels


def run(monitor, model, X, y, save):
    use_scaler = True
    trained_monitor = None
    layer_index = monitor.layer_index

    #building monitor with training set
    arrWeights, arrLabels = build_monitor(model, X, y, layer_index)
    #print("arrWeights:", np.shape(arrWeights))
    #print("arrLabels:", np.shape(arrLabels))
    if use_scaler:
        arrWeights = StandardScaler().fit_transform(arrWeights)

    if monitor.method == "knn":
        trained_monitor = KNeighborsClassifier(n_neighbors=monitor.n_clusters).fit(arrWeights, np.ravel(arrLabels))
    elif monitor.method == "hdbscan":
        trained_monitor = hdbscan.HDBSCAN(min_cluster_size=monitor.min_samples, prediction_data=True).fit(arrWeights)

    file_path = monitor.monitors_folder+monitor.filename
    if save:
        print("Saving monitor in", file_path)
        os.makedirs(monitor.monitors_folder, exist_ok=True)
        pickle.dump(trained_monitor, open( file_path, "wb" ))

    return trained_monitor