import os
import numpy as np
import pickle
import psutil
from src.utils import util
from keras.models import load_model


def run(X_test, y_test, model_build, monitor):
    arrPred = []
    models = []
    arrWeights = {}
    boxes = []
    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    classToMonitor = monitor.classToMonitor
    arrFalseNegative = {classToMonitor: 0}
    arrTrueNegative = {classToMonitor: 0}
    arrFalsePositive = {classToMonitor: 0}
    arrTruePositive = {classToMonitor: 0}

    # loading CNN trained with pre-processed images and preparing to store their weights
    for i in range(model_build.num_cnn):
        models.append(load_model(model_build.models_folder+model_build.model_name+str(i)+'.h5'))
        arrWeights.update({i: []})
        box = pickle.load(open(monitor.monitors_folder+monitor.monitor_name+str(i)+".p", "rb"))
        boxes.append(box[classToMonitor])

    dim_reduc_obj = None
    
    if monitor.dim_reduc_method != None:
        dim_reduc_obj = pickle.load(open(monitor.monitors_folder+monitor.dim_reduc_method, "rb"))
    
    #memory
    process = psutil.Process(os.getpid())

    for img, lab in zip(X_test, y_test):
        #counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        sequence_models = (models[i].predict(img)[0] for i in range(model_build.num_cnn))
        #aplying ensemble
        y_all = np.vstack(sequence_models)
        
        y_all = np.average(y_all, axis=0)
        yPred = np.argmax(y_all)
        arrPred.append(yPred)
        seq_interm_vals = (util.get_activ_func(models[i], img, layerIndex=monitor.layer_index)[0] for i in range(model_build.num_cnn))
        intermediateValues_all = np.vstack(seq_interm_vals)

        if yPred == classToMonitor:
            if monitor.method(boxes, intermediateValues_all, dim_reduc_obj):
                
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