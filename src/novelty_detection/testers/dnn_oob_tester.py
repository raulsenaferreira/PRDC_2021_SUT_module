import os
import numpy as np
import pickle
import psutil
from src.utils import util


def run(X_test, y_test, model, monitor, dataset_name):
    arrPred = []
    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    class_to_monitor = monitor.class_to_monitor
    arrFalseNegative = {class_to_monitor: 0}
    arrTrueNegative = {class_to_monitor: 0}
    arrFalsePositive = {class_to_monitor: 0}
    arrTruePositive = {class_to_monitor: 0}

    # loading abstraction boxes
    boxes=""
    try:
        monitor_path = monitor.monitors_folder+monitor.monitor_name+'_class_'
        monitor_path+=str(monitor.class_to_monitor)+'_'+dataset_name+".p"
        boxes = pickle.load(open(monitor_path, "rb"))
    except:
        print("Error while trying to open {} monitor!!!".format(monitor_path))
        return "","","","","","",""
    
    #memory
    process = psutil.Process(os.getpid())

    for img, lbl in zip(X_test, y_test):
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)
        intermediateValues = util.get_activ_func(model, img, monitor.layer_index)[0]

        if yPred == class_to_monitor:
            if monitor.method(boxes, intermediateValues, yPred, monitor.dim_reduc_method):
                
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