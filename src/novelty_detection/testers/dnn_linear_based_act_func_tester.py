import os
import numpy as np
import pickle
import psutil
from src.utils import util
from src.utils import safety_approaches
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
    monitor.arr_pred_monitor_ID, monitor.arr_lbl_monitor_ID = [], []
    monitor.arr_pred_monitor_OOD, monitor.arr_lbl_monitor_OOD = [], []

    monitor.arrFalseNegative_ID = {}
    monitor.arrTrueNegative_ID = {}
    monitor.arrFalsePositive_ID = {}
    monitor.arrTruePositive_ID = {}

    monitor.arrFalseNegative_OOD = {}
    monitor.arrTruePositive_OOD = {}

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    for class_to_monitor in range(experiment.classes_to_monitor_ID):
        # ID
        monitor.arrFalseNegative_ID.update({class_to_monitor: []})
        monitor.arrTrueNegative_ID.update({class_to_monitor: []})
        monitor.arrFalsePositive_ID.update({class_to_monitor: []})
        monitor.arrTruePositive_ID.update({class_to_monitor: []})

    for class_OOD in range(experiment.classes_to_monitor_ID, experiment.classes_to_monitor_OOD + experiment.classes_to_monitor_ID):
        # OOD
        monitor.arrFalseNegative_OOD.update({class_OOD: []})
        monitor.arrTruePositive_OOD.update({class_OOD: []})
    
    model = experiment.model

    #memory
    process = psutil.Process(os.getpid())

    # if you want to scale act func values 
    scaler = None
    if monitor.use_scaler:
        scaler_file = monitor.monitors_folder+'saved_scaler_'+monitor.filename
        scaler = pickle.load(open(scaler_file, "rb"))
        monitor.filename = '(scaled_input_version)'+monitor.filename

    # loading cluster-baed monitor
    monitor_path = monitor.monitors_folder +sep+ monitor.filename
    linear_based_monitor = pickle.load(open(monitor_path, "rb"))

    for img, lbl in zip(X_test, y_test):
                
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)

        intermediateValues = util.get_activ_func(model, img, monitor.layer_index)[0]
        intermediateValues = np.reshape(intermediateValues, (1, -1))
        
        monitor = safety_approaches.safety_monitor_decision(monitor, yPred, lbl, experiment.classes_to_monitor_ID,
         intermediateValues, scaler, linear_based_monitor)           

    memory = process.memory_info().rss / 1024 / 1024

    
    return arrPred, y_test, monitor.arr_pred_monitor_ID, monitor.arr_lbl_monitor_ID,\
     monitor.arr_pred_monitor_OOD, monitor.arr_lbl_monitor_OOD, memory, monitor.arrFalsePositive_ID, \
     monitor.arrFalseNegative_ID, monitor.arrTruePositive_ID, monitor.arrTrueNegative_ID, \
     monitor.arrFalseNegative_OOD, monitor.arrTruePositive_OOD