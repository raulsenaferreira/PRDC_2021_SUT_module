import os
import numpy as np
import pickle
import psutil
from src.utils import util
import matplotlib.pyplot as plt
from src.novelty_detection.methods import image_dist_matching as idm


sep = util.get_separator()


def run(X_test, y_test, experiment, monitor, dataset_name):
    arrPred = []
    arrFalseNegative_ID = {}
    arrTrueNegative_ID = {}
    arrFalsePositive_ID = {}
    arrTruePositive_ID = {}

    arrFalseNegative_OOD = {}
    arrTruePositive_OOD = {}

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    for class_to_monitor in range(experiment.classes_to_monitor):
        # ID
        arrFalseNegative_ID.update({class_to_monitor: 0})
        arrTrueNegative_ID.update({class_to_monitor: 0})
        arrFalsePositive_ID.update({class_to_monitor: 0})
        arrTruePositive_ID.update({class_to_monitor: 0})
        # OOD
        arrFalseNegative_OOD.update({class_to_monitor: []})
        arrTruePositive_OOD.update({class_to_monitor: []})
    
    model = experiment.model

    zeros = 0
    #memory
    process = psutil.Process(os.getpid())

    counter_monitor = 0
    missclassified_images_monitor = []
    missclassified_image_labels_monitor = []
    missclassified_images_monitor_similarity = []

    counter_monitor_TP = 0
    TP_classified_images_monitor = []
    TP_image_labels_monitor = []

    counter_DNN = 0
    missclassified_images_DNN = []
    missclassified_image_labels_DNN = []

    for img, lbl in zip(X_test, y_test):
                
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)
        intermediateValues = util.get_activ_func(model, img, monitor.layer_index)[0]
        
        # loading abstraction boxes
        monitor_path = monitor.monitors_folder+str(yPred) +sep+ monitor.filename
        boxes = pickle.load(open(monitor_path, "rb"))
        #print(np.shape(boxes))
        
        is_in_the_box = monitor.method(boxes, intermediateValues, yPred, monitor.monitors_folder, monitor.dim_reduc_method)
        zeros+=is_in_the_box[1]
        
        if lbl < experiment.classes_to_monitor: # OOD label numbers starts after the ID label numbers
            if is_in_the_box[0]:
                if yPred != lbl:
                    arrFalseNegative_ID[yPred] += 1 #False negative 
                             
                if yPred == lbl: 
                    arrTrueNegative_ID[yPred] += 1 #True negatives
            else:
                if yPred != lbl: 
                    arrTruePositive_ID[yPred] += 1 #True positives
                    
                if yPred == lbl: 
                    arrFalsePositive_ID[yPred] += 1 #False positives
                    
        else:
            if is_in_the_box[0]:
                arrFalseNegative_OOD[yPred].append(lbl) #False negative           
                
            else:
                arrTruePositive_OOD[yPred].append(lbl) #True positives
        
    #print("zeroed points:", zeros)
    memory = process.memory_info().rss / 1024 / 1024

    return arrPred, y_test, memory, arrFalsePositive_ID, arrFalseNegative_ID, arrTruePositive_ID, arrTrueNegative_ID, arrFalseNegative_OOD, arrTruePositive_OOD