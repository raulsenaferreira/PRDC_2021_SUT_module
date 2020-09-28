import os
import numpy as np
import pickle
import psutil
from src.utils import util
from src.utils import safety_approaches
from src.Classes.readout import Readout
import matplotlib.pyplot as plt



sep = util.get_separator()
        

def run(dataset, experiment, monitor):
    # special case when GTSRB + BTSC due to the intersection between some classes
    monitor.map_dataset_classes = True if experiment.dataset.dataset_ID_name == 'GTSRB' and experiment.dataset.dataset_OOD_name == 'BTSC' else False

    X_test, y_test = dataset.X, dataset.y
    dataset_name = dataset.dataset_name
    arrPred = []
    loaded_monitor = {}

    readout = Readout()

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    for class_to_monitor in range(experiment.classes_to_monitor_ID):
        # ID
        readout.arr_false_negative_ID.update({class_to_monitor: []})
        readout.arr_true_negative_ID.update({class_to_monitor: []})
        readout.arr_false_positive_ID.update({class_to_monitor: []})
        readout.arr_true_positive_ID.update({class_to_monitor: []})

    for class_OOD in range(experiment.classes_to_monitor_ID, experiment.classes_to_monitor_OOD + experiment.classes_to_monitor_ID):
        # OOD
        readout.arr_false_negative_OOD.update({class_OOD: []})
        readout.arr_true_negative_OOD.update({class_OOD: []})
        readout.arr_false_positive_OOD.update({class_OOD: []})
        readout.arr_true_positive_OOD.update({class_OOD: []})
    
    model = experiment.model

    #memory
    process = psutil.Process(os.getpid())

    # if you want to scale act func values 
    scaler = None
    if monitor.use_scaler:
        scaler_file = monitor.monitors_folder+'saved_scaler_'+monitor.filename
        scaler = pickle.load(open(scaler_file, "rb"))
        monitor.filename = '(scaled_input_version)'+monitor.filename

    if monitor.OOD_approach == 'outside_of_box':
        for c in range(experiment.classes_to_monitor_ID): 
            monitor_path = os.path.join(monitor.monitors_folder+str(c), monitor.filename)
            loaded_monitor.update({c: pickle.load(open(monitor_path, "rb"))})
    else:
        monitor_path = os.path.join(monitor.monitors_folder, monitor.filename)
        loaded_monitor = pickle.load(open(monitor_path, "rb"))

    for img, lbl in zip(X_test, y_test):
        if experiment.verbose:
            counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)

        intermediateValues = util.get_activ_func(model, img, monitor.layer_index)[0]
        
        readout = safety_approaches.safety_monitor_decision(readout, monitor, yPred, lbl, experiment.classes_to_monitor_ID,
         intermediateValues, scaler, loaded_monitor)           

    readout.memory = process.memory_info().rss / 1024 / 1024
    
    return arrPred, y_test, readout