import os
import numpy as np
import pickle
import psutil
from src.utils import util


def run(X_test, y_test, experiment, monitor, dataset_name):
    arrPred = []
    arrFalseNegative = {}
    arrTrueNegative = {}
    arrFalsePositive = {}
    arrTruePositive = {}

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    for class_to_monitor in range(experiment.classes_to_monitor):
        arrFalseNegative.update({class_to_monitor: 0})
        arrTrueNegative.update({class_to_monitor: 0})
        arrFalsePositive.update({class_to_monitor: 0})
        arrTruePositive.update({class_to_monitor: 0})
    
    model = experiment.model

    #memory
    process = psutil.Process(os.getpid())

    for img, lbl in zip(X_test, y_test):
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)
        intermediateValues = util.get_activ_func(model, img, monitor.layer_index)[0]
        
        #for class_to_monitor in range(experiment.classes_to_monitor): 
            #if yPred == class_to_monitor:
        if monitor.method(monitor.boxes, intermediateValues, yPred, monitor.dim_reduc_method):    
            if yPred != lbl:
                arrFalseNegative[class_to_monitor] += 1 #False negative           
            if yPred == lbl: 
                arrTrueNegative[class_to_monitor] += 1 #True negatives
        else:
            if yPred != lbl: 
                arrTruePositive[class_to_monitor] += 1 #True positives
            if yPred == lbl: 
                arrFalsePositive[class_to_monitor] += 1 #False positives
                    
        #elif lbl==class_to_monitor and yPred != class_to_monitor:
            #print("missclassification --- new pattern for class",yPred, str(lbl))
    memory = process.memory_info().rss / 1024 / 1024

    return arrPred, y_test, memory, arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative