import numpy as np
import pandas as pd
import pickle
from src.utils import util
from src.utils import abstraction_box
from keras.models import load_model


def run(classToMonitor, layer_name, models_folder, monitors_folder, isTestOneClass, sep, script_path):
    count = [0, 0]
    arrPred = []
    num_classes = 43
    testPath = str(script_path)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep
    #loading test set
    X_test, y_test = util.load_GTRSB_csv(testPath, "Test.csv")

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    # loading model and abstraction boxes
    model = load_model(models_folder+'CNN_GTRSB.h5')
    boxes = pickle.load(open(monitors_folder+"monitor_Box_GTRSB.p", "rb")) 

    arrFalseNegative = {str(classToMonitor): 0}
    arrTrueNegative = {str(classToMonitor): 0}
    arrFalsePositive = {str(classToMonitor): 0}
    arrTruePositive = {str(classToMonitor): 0}

    for img, lab in zip(X_test, y_test):
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        yPred = np.argmax(model.predict(img))
        arrPred.append(yPred)
        intermediateValues = util.get_activ_func(model, img, layerName=layer_name)[0]

        if abstraction_box.find_point(boxes, intermediateValues, yPred):
            count[0] += 1
            if yPred != lab:
                arrFalseNegative[str(classToMonitor)] += 1 #False negative			
            if yPred == lab: 
                arrTrueNegative[str(classToMonitor)] += 1 #True negatives
        else:
            if yPred == classToMonitor:
                count[1] += 1
                if yPred != lab: 
                    arrTruePositive[str(classToMonitor)] += 1 #True positives
                if yPred == lab: 
                    arrFalsePositive[str(classToMonitor)] += 1 #False positives
            #elif lab==classToMonitor:
                #print("missclassification --- new pattern for class",yPred, str(lab))
    return count, arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative