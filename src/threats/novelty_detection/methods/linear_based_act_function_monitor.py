import os
import numpy as np
import pickle
from src.utils import util
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
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
    text_best_params = "best_params"
    arrWeights, arrLabels = None, None
    scaler = None
    trained_monitor = None
    layer_index = monitor.layer_index

    #building monitor with training set
    if monitor.use_alternative_monitor:
        arrWeights, arrLabels = build_monitor_2(model, X, y, layer_index)
    else:
        arrWeights, arrLabels = build_monitor(model, X, y, layer_index)
    #print("arrLabels:", np.shape(arrLabels))
    if monitor.use_scaler:
        scaler = StandardScaler().fit(arrWeights)
        scaler_file = monitor.monitors_folder+'saved_scaler_'+monitor.filename

        print("Saving standard scaler object in", scaler_file)
        os.makedirs(monitor.monitors_folder, exist_ok=True)
        pickle.dump(scaler, open( scaler_file, "wb" ))

        arrWeights = scaler.transform(arrWeights)
        monitor.filename = '(scaled_input_version)'+monitor.filename
        text_best_params = '(scaled_input_version)'+text_best_params

    if monitor.method == "sgd":

        sgdc=SGDClassifier(random_state=42)

        if monitor.use_grid_search:
            print("optimizing with Grid Search")
            param_grid = { 
                'max_iter': [1000, 2000],
                'alpha': [0.0001, 0.001],
                'loss': ['squared_hinge', 'log', 'perceptron'],
                'class_weight' :['balanced', None]
            }
            CV_sgdc = GridSearchCV(estimator=sgdc, param_grid=param_grid, cv= 5)
            CV_sgdc.fit(arrWeights, np.ravel(arrLabels))
            trained_monitor = CV_sgdc.best_estimator_

            if monitor.use_alternative_monitor:
                text_best_params += "_2.txt"
            else:
                text_best_params += ".txt"

            file = open(monitor.monitors_folder+text_best_params,"w")#write mode 
            file.write(str(CV_sgdc.best_params_)) 
            file.close()             

        else:
            trained_monitor = sgdc.fit(arrWeights, np.ravel(arrLabels))
        
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