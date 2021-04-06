import os
import numpy as np
import pickle
import psutil
from src.utils import util
from src.threats.novelty_detection.utils import safety_approaches as SA# use this for novelty detection only
#from src.threats.novelty_detection.utils import safety_approaches_2 as SA
from src.Classes.readout import Readout
import matplotlib.pyplot as plt
from time import perf_counter as timer

        

def run(dataset, experiment, monitor):
    
    X_test, y_test = dataset.X, dataset.y
    dataset_name = dataset.dataset_name
    #print(len(y_test))
    #print('ID:', len(np.where(y_test<43)[0]))
    #print('OOD:', len(np.where(y_test>=43)[0]))
    loaded_monitor = {}

    readout = Readout()

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))
    
    model = experiment.model

    arr_ml_time = []
    arr_sm_time = []

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
    elif monitor.OOD_approach == 'temperature' or monitor.OOD_approach == 'adversarial':
        loaded_monitor = monitor.method
    else:
        monitor_path = os.path.join(monitor.monitors_folder, monitor.filename)
        loaded_monitor = pickle.load(open(monitor_path, "rb"))

    for img, lbl in zip(X_test, y_test):
        if experiment.verbose:
            counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        
        img = np.asarray([img])

        ini_ml = timer()
        yPred = np.argmax(model.predict(img))
        end_ml = timer()

        # ML readout
        readout.arr_classification_pred.append(yPred)
        arr_ml_time.append(end_ml-ini_ml)

        # SM readout
        if monitor.OOD_approach == 'outside_of_box':
            use_intermediateValues = True
        
        elif monitor.OOD_approach == 'temperature' or monitor.OOD_approach == 'adversarial':
            use_intermediateValues = False

        readout, time_spent = SA.safety_monitor_decision(readout, monitor, model, img, yPred, lbl, experiment,
         use_intermediateValues, scaler, loaded_monitor)
        
        arr_sm_time.append(time_spent)

    # some complementaire general readout
    readout.total_memory = process.memory_info().rss / 1024 / 1024

    # some complementaire ML readout
    readout.arr_classification_true = y_test
    readout.ML_time = arr_ml_time

    # some complementaire SM readout
    readout.SM_time = arr_sm_time
    
    return readout