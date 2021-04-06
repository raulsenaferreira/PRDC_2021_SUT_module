import os
import numpy as np
import pickle
import psutil
from src.utils import util
#from src.threats.novelty_detection.utils import safety_approaches # use this just for novelty detection
from src.threats.novelty_detection.utils import safety_approaches_2
from src.Classes.readout import Readout
import matplotlib.pyplot as plt
from time import perf_counter as timer
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

        

def run(dataset, experiment, monitor):
    # special case when GTSRB + BTSC due to the intersection between some classes
    monitor.map_dataset_classes = True if experiment.dataset.modification == 'gtsrb_btsc' else False

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

    if monitor.OOD_approach == 'temperature':
        loaded_monitor = monitor.method
    else:
        monitor_path = os.path.join(monitor.monitors_folder, monitor.filename)
        loaded_monitor = pickle.load(open(monitor_path, "rb"))

    #for img, lbl in zip(X_test, y_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    model.to(device)
    cudnn.benchmark = True

    #testset = 
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=0)

    #with torch.no_grad():
    for data in testloader:
        if experiment.verbose:
            counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        
        img, lbl = data[0].to(device, dtype=torch.float), data[1].to(device)
        #print(np.shape(img))

        ini_ml = timer()
        outputs = model(img)
        _, yPred = torch.max(outputs.data, 1)
        end_ml = timer()
        #print(yPred)

        # ML readout
        readout.arr_classification_pred.append(yPred)
        arr_ml_time.append(end_ml-ini_ml)

        # SM readout        
        if monitor.OOD_approach == 'temperature':
            use_intermediateValues = False

        readout, time_spent = safety_approaches_2.safety_monitor_decision(readout, monitor, model, img, yPred, lbl, experiment.classes_to_monitor_ID,
         use_intermediateValues, None, loaded_monitor)
        
        arr_sm_time.append(time_spent)

    # some complementaire general readout
    readout.total_memory = process.memory_info().rss / 1024 / 1024

    # some complementaire ML readout
    readout.arr_classification_true = y_test
    readout.ML_time = arr_ml_time

    # some complementaire SM readout
    readout.SM_time = arr_sm_time
    
    return readout