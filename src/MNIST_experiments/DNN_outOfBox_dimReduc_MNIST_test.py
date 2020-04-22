import os
import psutil
import numpy as np
import pandas as pd
import pickle
from src.utils import util
from src.utils import abstraction_box
from keras.models import load_model


def run(classToMonitor, layer_name, models_folder, monitors_folder, monitor_name, model_name, isTestOneClass, sep, dim_reduc_method):
    count = [0, 0]
    arrPred = []
    has_known_pattern = None

    #loading test set
    _, _, X_test, y_test, _ = util.load_mnist(onehotencoder=False)

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    # loading model, abstraction boxes, and the trained dim reduction method if it exists
    model = load_model(models_folder+model_name)
    boxes = pickle.load(open(monitors_folder+monitor_name, "rb"))
    dim_reduc_obj = None
    
    if dim_reduc_method != None:
        dim_reduc_obj = pickle.load(open(monitors_folder+dim_reduc_method+'_MNIST_trained.p', "rb"))

    arrFalseNegative = {str(classToMonitor): 0}
    arrTrueNegative = {str(classToMonitor): 0}
    arrFalsePositive = {str(classToMonitor): 0}
    arrTruePositive = {str(classToMonitor): 0}

    for img, lab in zip(X_test, y_test):
        #memory
        process = psutil.Process(os.getpid())

        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)
        intermediateValues = util.get_activ_func(model, img, layerName=layer_name)[0]

        if dim_reduc_method != None:
            has_known_pattern = abstraction_box.find_point(boxes, intermediateValues, yPred, dim_reduc_obj)
        else:
            has_known_pattern = abstraction_box.find_point(boxes, intermediateValues, yPred)

        if yPred == classToMonitor:
            if has_known_pattern:
                count[0] += 1
                if yPred != lab:
                    arrFalseNegative[str(classToMonitor)] += 1 #False negative          
                if yPred == lab: 
                    arrTrueNegative[str(classToMonitor)] += 1 #True negatives
            else:
                count[1] += 1
                if yPred != lab: 
                    arrTruePositive[str(classToMonitor)] += 1 #True positives
                if yPred == lab: 
                    arrFalsePositive[str(classToMonitor)] += 1 #False positives
        memory = process.memory_info().rss / 1024 / 1024
        #elif lab==classToMonitor and yPred != classToMonitor:
            #print("missclassification --- new pattern for class",yPred, str(lab))
    return arrPred, y_test, memory, arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative