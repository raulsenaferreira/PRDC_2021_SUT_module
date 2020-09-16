import os
import numpy as np
import pickle
import psutil
from src.utils import util
import matplotlib.pyplot as plt



sep = util.get_separator()


def plot_images(title, data, labels, similarities, num_row, num_col):

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        try:
            ax = axes[i//num_col, i%num_col]
            ax.imshow(np.squeeze(data[i]), cmap='gray')
            ax.set_title('{}-Sim={}'.format(labels[i], similarities[i]))
            ax.set_axis_off()
        except Exception as e:
            pass    
    fig.suptitle(title)    
    plt.tight_layout(pad=3.0)
    plt.show()


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

    # loading cluster-baed monitor
    monitor_path = monitor.monitors_folder +sep+ monitor.filename
    cluster_based_monitor = pickle.load(open(monitor_path, "rb"))

    for img, lbl in zip(X_test, y_test):
                
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)
        intermediateValues = util.get_activ_func(model, img, monitor.layer_index)[0]
        
        yPred_by_monitor = cluster_based_monitor.predict(np.reshape(intermediateValues, (1, -1)))
        #print(np.shape(yPred_by_monitor))
        
        if lbl < experiment.classes_to_monitor: # OOD label numbers starts after the ID label numbers
            if yPred_by_monitor == yPred:
                if yPred != lbl:
                    arrFalseNegative_ID[yPred] += 1 #False negative 
                    #counter_DNN+=1
                    #missclassified_images_DNN.append(img)
                    #missclassified_image_labels_DNN.append(yPred)          
                if yPred == lbl: 
                    arrTrueNegative_ID[yPred] += 1 #True negatives
            else:
                if yPred != lbl: 
                    arrTruePositive_ID[yPred] += 1 #True positives
                    #counter_monitor_TP+=1
                    #TP_classified_images_monitor.append(img)
                    #TP_image_labels_monitor.append(yPred)
                if yPred == lbl: 
                    arrFalsePositive_ID[yPred] += 1 #False positives
                    #counter_monitor+=1
                    #missclassified_images_monitor.append(img)
                    #missclassified_image_labels_monitor.append(yPred)
                    
        else:
            if yPred_by_monitor == yPred:
                arrFalseNegative_OOD[yPred].append(lbl) #False negative           
                #if yPred == lbl: 
                    #arrTrueNegative_OOD[yPred] += 1 #True negatives
            else:
                arrTruePositive_OOD[yPred].append(lbl) #True positives
                #if yPred == lbl: 
                    #arrFalsePositive_OOD[yPred] += 1 #False positives
        '''
        if counter_monitor_TP % 10 == 0 and counter_monitor_TP > 0:
            plot_images("True positives (Monitor detected right)", TP_classified_images_monitor, TP_image_labels_monitor, 2, 5)
            counter_monitor_TP = 0
            TP_classified_images_monitor = []
            TP_image_labels_monitor = []
        
        if counter_monitor % 60 == 0 and counter_monitor > 0:
            plot_images("False positives (Monitor misclassified)", missclassified_images_monitor, missclassified_image_labels_monitor, missclassified_images_monitor_similarity, 6, 10)
            counter_monitor = 0
            missclassified_images_monitor = []
            missclassified_image_labels_monitor = []
        
        if counter_DNN % 60 == 0 and counter_DNN > 0:
            plot_images("False negatives (DNN misclassified)", missclassified_images_DNN, missclassified_image_labels_DNN, 6, 10)
            counter_DNN = 0
            missclassified_images_DNN = []
            missclassified_image_labels_DNN = []
        '''

    #print("zeroed points:", zeros)
    memory = process.memory_info().rss / 1024 / 1024

    return arrPred, y_test, memory, arrFalsePositive_ID, arrFalseNegative_ID, arrTruePositive_ID, arrTrueNegative_ID, arrFalseNegative_OOD, arrTruePositive_OOD