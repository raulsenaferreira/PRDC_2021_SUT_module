import numpy as np
import pickle
from src.utils import util
from src.utils import abstraction_box
from keras.models import load_model


def run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, sep, script_path):
    count = [0, 0]
    arrPred = []
    models = []
    arrWeights = {}
    boxes = []

    arrFalseNegative = {str(classToMonitor): 0}
    arrTrueNegative = {str(classToMonitor): 0}
    arrFalsePositive = {str(classToMonitor): 0}
    arrTruePositive = {str(classToMonitor): 0}
    #loading test set
    testPath = str(script_path)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep
    X_test, y_test = util.load_GTRSB_csv(testPath, "Test.csv")

    #3 variables for log (optional)
    counter = 0
    loading_percentage = 0.1
    loaded = int(loading_percentage*len(y_test))

    # adding CNN trained with pre-processed images and preparing to store their weights
    for i in range(num_cnn):
        models.append(load_model(models_folder+model_ensemble_prefix+str(i)+'.h5'))
        arrWeights.update({i: []})
        box = pickle.load(open(monitors_ensemble_folder+monitor_ensemble_prefix+str(i)+".p", "rb"))
        boxes.append(box[classToMonitor])
    
    for img, lab in zip(X_test, y_test):
        lab = np.where(lab)[0]
        counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
        img = np.asarray([img])
        #aplying ensemble
        y_all = np.vstack((
            models[0].predict(img)[0], models[1].predict(img)[0], models[2].predict(img)[0], 
            models[3].predict(img)[0], models[4].predict(img)[0]
            ))
        
        y_all = np.average(y_all, axis=0)
        yPred = np.argmax(y_all)
        arrPred.append(yPred)

        intermediateValues_all = np.vstack((
            util.get_activ_func(models[0], img, layerIndex=layer_index)[0], util.get_activ_func(models[1], img, layerIndex=layer_index)[0], 
            util.get_activ_func(models[2], img, layerIndex=layer_index)[0], util.get_activ_func(models[3], img, layerIndex=layer_index)[0], 
            util.get_activ_func(models[4], img, layerIndex=layer_index)[0]
            ))

        if yPred == classToMonitor:
            if abstraction_box.find_point_box_ensemble(boxes, intermediateValues_all):
                count[0] += 1
                if yPred != lab:
                    arrFalseNegative[str(classToMonitor)] += 1 #False negative          
                if yPred == lab: 
                    arrTrueNegative[str(classToMonitor)] += 1 #True negatives
            else:
                count[1] += 1
                if yPred != lab: 
                    arrTruePositive[str(classToMonitor)] += 1 #True positives
                if yPred == lab: 
                    arrFalsePositive[str(classToMonitor)] += 1 #False positives
        #elif lab==classToMonitor and yPred != classToMonitor:
            #print("missclassification --- new pattern for class",yPred, str(lab))

    return arrPred, count, arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative