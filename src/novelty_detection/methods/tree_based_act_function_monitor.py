import os
import numpy as np
import pickle
from src.utils import util
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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


def build_monitor_2(model, X, y, layer_index):
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
        
        arrWeights.append(util.get_activ_func(model, img, layerIndex=layer_index)[0])
        arrLabels.append(lab)

    return arrWeights, arrLabels


def run(monitor, model, X, y, save):
    optimize_parameters = monitor.use_grid_search
    trained_monitor = None
    layer_index = monitor.layer_index

    #building monitor with training set
    if monitor.use_alternative_monitor:
        arrWeights, arrLabels = build_monitor_2(model, X, y, layer_index)
    else:
        arrWeights, arrLabels = build_monitor(model, X, y, layer_index)
    #print("arrWeights:", np.shape(arrWeights))
    #print("arrLabels:", np.shape(arrLabels))

    if monitor.method == "random_forest":
        rfc=RandomForestClassifier(random_state=42)

        if optimize_parameters:
            param_grid = { 
                'n_estimators': [100, 200],
                'max_features': ['sqrt', 'log2'],
                'criterion' :['gini', 'entropy']
            }
            CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
            CV_rfc.fit(arrWeights, np.ravel(arrLabels))
            trained_monitor = CV_rfc.best_estimator_

            file1 = open(monitor.monitors_folder+"best_params.txt","w")#write mode 
            file1.write(str(CV_rfc.best_params_)) 
            file1.close()

        else:
            trained_monitor = rfc.fit(arrWeights, np.ravel(arrLabels))
        
    elif monitor.method == "":
        pass

    file_path = None
    if monitor.use_alternative_monitor:
        file_path = monitor.monitors_folder+monitor.filename+'_2'
    else:
        file_path = monitor.monitors_folder+monitor.filename

    if save:
        print("Saving monitor in", file_path)
        os.makedirs(monitor.monitors_folder, exist_ok=True)
        pickle.dump(trained_monitor, open( file_path, "wb" ))

    return trained_monitor