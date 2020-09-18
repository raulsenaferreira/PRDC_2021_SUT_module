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


def run(monitor, model, X, y, save):
    use_scaler = True
    optimize_parameters = monitor.use_grid_search
    trained_monitor = None
    layer_index = monitor.layer_index

    #building monitor with training set
    arrWeights, arrLabels = build_monitor(model, X, y, layer_index)
    #print("arrWeights:", np.shape(arrWeights))
    #print("arrLabels:", np.shape(arrLabels))
    if use_scaler:
        arrWeights = StandardScaler().fit_transform(arrWeights)

    if monitor.method == "sgd":
        sgdc=SGDClassifier(random_state=42)

        if optimize_parameters:
            param_grid = { 
                'alpha': [0.0001, 0.001],
                'loss': ['squared_hinge', 'log', 'perceptron'],
                'class_weight' :['balanced', None]
            }
            CV_sgdc = GridSearchCV(estimator=sgdc, param_grid=param_grid, cv= 5)
            CV_sgdc.fit(arrWeights, np.ravel(arrLabels))
            trained_monitor = CV_sgdc.best_estimator_

        else:
            trained_monitor = sgdc.fit(arrWeights, np.ravel(arrLabels))
        
    elif monitor.method == "":
        pass

    file_path = monitor.monitors_folder+monitor.filename
    if save:
        print("Saving monitor in", file_path)
        os.makedirs(monitor.monitors_folder, exist_ok=True)
        pickle.dump(trained_monitor, open( file_path, "wb" ))

    return trained_monitor