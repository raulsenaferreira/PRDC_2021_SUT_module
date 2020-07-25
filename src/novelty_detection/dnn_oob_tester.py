import os
import numpy as np
import pickle
import psutil
from src.utils import util


def run(X_test, y_test, model, monitor):
    arrPred = []
    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    classToMonitor = str(monitor.classToMonitor)
    arrFalseNegative = {classToMonitor: 0}
    arrTrueNegative = {classToMonitor: 0}
    arrFalsePositive = {classToMonitor: 0}
    arrTruePositive = {classToMonitor: 0}

    # loading abstraction boxes
    boxes = pickle.load(open(monitor.monitors_folder+monitor.monitor_name, "rb"))
    dim_reduc_obj = None
    
    if monitor.dim_reduc_method != None:
        dim_reduc_obj = pickle.load(open(monitor.monitors_folder+monitor.dim_reduc_method, "rb"))
    #memory
    process = psutil.Process(os.getpid())

    for img, lab in zip(X_test, y_test):
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)
        intermediateValues = util.get_activ_func(model, img, monitor.layer_index)[0]

        if yPred == monitor.classToMonitor:
            if monitor.method(boxes, intermediateValues, yPred, dim_reduc_obj):
                
                if yPred != lab:
                    arrFalseNegative[classToMonitor] += 1 #False negative           
                if yPred == lab: 
                    arrTrueNegative[classToMonitor] += 1 #True negatives
            else:
                if yPred != lab: 
                    arrTruePositive[classToMonitor] += 1 #True positives
                if yPred == lab: 
                    arrFalsePositive[classToMonitor] += 1 #False positives
                    
        #elif lab==classToMonitor and yPred != classToMonitor:
            #print("missclassification --- new pattern for class",yPred, str(lab))
    memory = process.memory_info().rss / 1024 / 1024

    return arrPred, y_test, memory, arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative