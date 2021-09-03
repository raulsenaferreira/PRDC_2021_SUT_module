import os
import pickle
from time import perf_counter as timer
from src.utils import util
from keras.losses import binary_crossentropy
import keras.backend as K
import numpy as np


def is_pred_diff(yPred, intermediateValues, loaded_monitor):
    yPred_by_monitor = loaded_monitor.predict(intermediateValues)
    #print(np.shape(yPred_by_monitor))

    if yPred_by_monitor == yPred:
        return False
    
    return True


def is_pred_neg(yPred, intermediateValues, loaded_monitor):
    yPred_by_monitor = loaded_monitor.predict(intermediateValues)
    #print(np.shape(yPred_by_monitor))

    if yPred_by_monitor == -1:
        return True
    
    return False


def safety_monitor_decision(readout, monitor, model, img, yPred, lbl, experiment, use_intermediateValues,
 scaler, loaded_monitor):
    
    classes_to_monitor = experiment.classes_to_monitor_ID
    raise_alarm = False

    ini = timer() # SM time

    if use_intermediateValues == True:
        intermediateValues = util.get_activ_func(experiment.backend, model, img, monitor.layer_index)[0]
        # if you want to scale act func values
        if scaler != None:
            intermediateValues = scaler.transform(intermediateValues)

    if monitor.OOD_approach == 'equality':
        # if monitor acceptance approach is based on two equal predictions
        intermediateValues = np.reshape(intermediateValues, (1, -1))
        raise_alarm = is_pred_diff(yPred, intermediateValues, loaded_monitor)

    elif monitor.OOD_approach == 'outlier':
        # if monitor acceptance approach is based on instance classified as outlier "-1"
        intermediateValues = np.reshape(intermediateValues, (1, -1))
        raise_alarm = is_pred_neg(yPred, intermediateValues, loaded_monitor)

    elif monitor.OOD_approach == 'outside_of_box':
        raise_alarm = monitor.method(loaded_monitor[yPred], intermediateValues, yPred, monitor)
    
    elif monitor.OOD_approach == 'temperature':
        # keras version
        #raise_alarm = loaded_monitor.detection(model, img, yPred, monitor.noiseMagnitude, monitor.temper, monitor.threshold)
        # pytorch version
        raise_alarm = loaded_monitor.detection(model, img, monitor.temper, monitor.noiseMagnitude, monitor.threshold, 'cuda:0')

    elif monitor.OOD_approach == 'adversarial':
        input_shape = np.shape(img) #(32, 32, 3)

        path = os.path.join(monitor.monitors_folder, 'class_{}'.format(yPred), 'ALOCC_Model_{}.h5'.format(monitor.model_number))

        monitor.method.adversarial_model.load_weights(path)
            
        #model_predicts = monitor.method.adversarial_model.predict(np.asarray([img]))
        model_predicts = monitor.method.adversarial_model.predict(img)

        input_image = img.reshape(input_shape)
        reconstructed_image = model_predicts[0].reshape(input_shape)

        y_true = K.variable(reconstructed_image)
        y_pred = K.variable(input_image)
        error = K.eval(binary_crossentropy(y_true, y_pred)).mean()

        if monitor.threshold[yPred] < error:
            raise_alarm = True

    # ID images (OOD label numbers higher than the ID label numbers)
    if lbl < classes_to_monitor: 

        # An ID image arrives in the stream, the SM raises the alarm and dismiss the ML classification
        if raise_alarm:    
            # false positive for OOD
            readout.arr_detection_SM.append(1)
            readout.arr_detection_true.append(0)
            
            if yPred != lbl: 
                # correct reaction (avoided a misclassification of ID data.)
                readout.arr_reaction_SM.append(1)
                readout.arr_reaction_true.append(1)
                
            if yPred == lbl: 
                # incorrect reaction (intervention with no necessity)
                readout.arr_reaction_SM.append(1)
                readout.arr_reaction_true.append(0)
        
        # An ID image arrives in the stream, the SM does not raise the alarm, accepting the ML classification
        else:
            # True negative for OOD
            readout.arr_detection_SM.append(0)
            readout.arr_detection_true.append(0)
            
            if yPred != lbl: 
                # incorrect reaction (it should intervene)
                readout.arr_reaction_SM.append(0)
                readout.arr_reaction_true.append(1)
                
            if yPred == lbl: 
                # correct reaction (it correctly did not intervene)
                readout.arr_reaction_SM.append(0)
                readout.arr_reaction_true.append(0)

    # OOD images
    else:
        
        # An OOD image arrives in the stream, the SM raises the alarm and dismiss the ML classification
        if raise_alarm:
            # True positive for OOD
            readout.arr_detection_SM.append(1)
            readout.arr_detection_true.append(1)

            # for novelty and outlier, it is always correct reaction (independently of the ground truth)
            readout.arr_reaction_SM.append(1)
            readout.arr_reaction_true.append(1)

        # An OOD image arrives in the stream, the SM does not raise the alarm, accepting the ML classification
        else: 
            # False negative for OOD
            readout.arr_detection_SM.append(0)
            readout.arr_detection_true.append(1)

            # for novelty and outlier, it is always incorrect reaction (it should intervene independently of the ground truth)
            readout.arr_reaction_SM.append(0)
            readout.arr_reaction_true.append(1)

    time_spent = timer() - ini
    return readout, time_spent