import os
import pickle
from time import perf_counter as timer
from src.utils import util


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


def map_btsc_gtsrb(y_gtsrb, y_btsc):
    # BTSC and GTSRB have 18 classes in common
    GTSRB_to_BTSC = {14:21, 22:0, 19:3, 20:4, 21:5, 25:10, 28:7, 26:11, 18:13,\
    24:16, 11:17, 13:19, 17:22, 15:28, 4:32, 35:34, 36:36, 12:61}

    try:
        if GTSRB_to_BTSC[y_gtsrb]+43 == y_btsc:
            return y_gtsrb
    except:
        return y_btsc

    return y_btsc


def safety_monitor_decision(readout, monitor, model, img, yPred, lbl, experiment, use_intermediateValues,
 scaler, loaded_monitor):
    
    classes_to_monitor = experiment.classes_to_monitor_ID
    raise_alarm = False

    # just when GTSRB = ID and BTSC = OOD
    if monitor.map_dataset_classes:
        #print('doing mapping between gtsrb and btsc ...')
        lbl = map_btsc_gtsrb(yPred, lbl)

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
        # getting the original OOD labels before the transformation (except for novelty detection datasets)
        original_lbl = lbl-classes_to_monitor

        # An OOD image arrives in the stream, the SM raises the alarm and dismiss the ML classification
        if raise_alarm:
            # True positive for OOD
            readout.arr_detection_SM.append(1)
            readout.arr_detection_true.append(1)

            if yPred != original_lbl:
                # correct reaction (avoided a misclassification of OOD data.) 
                readout.arr_reaction_SM.append(1)
                readout.arr_reaction_true.append(1)

            if yPred == original_lbl:
                # incorrect reaction (intervention with no necessity)
                readout.arr_reaction_SM.append(1)
                readout.arr_reaction_true.append(0)
                
        # An OOD image arrives in the stream, the SM does not raise the alarm, accepting the ML classification
        else: 
            # False negative for OOD
            readout.arr_detection_SM.append(0)
            readout.arr_detection_true.append(1)

            if yPred != original_lbl:
                # incorrect reaction (it should intervene)
                readout.arr_reaction_SM.append(0)
                readout.arr_reaction_true.append(1)

            if yPred == original_lbl:
                # correct reaction (it correctly did not intervene)
                readout.arr_reaction_SM.append(0)
                readout.arr_reaction_true.append(0)

    time_spent = timer() - ini

    return readout, time_spent