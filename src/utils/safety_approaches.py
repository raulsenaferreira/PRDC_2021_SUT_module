
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

    if GTSRB_to_BTSC[y_gtsrb]+43 == y_btsc:
        return y_gtsrb

    return y_btsc


def safety_monitor_decision(monitor, yPred, lbl, classes_to_monitor, intermediateValues,
 scaler, loaded_monitor):
    
    # if you want to scale act func values
    if scaler != None:
        intermediateValues = scaler.transform(intermediateValues)

    raise_alarm = False

    if monitor.OOD_approach == 'equality':
        # if monitor acceptance approach is based on two equal predictions
        raise_alarm = is_pred_diff(yPred, intermediateValues, loaded_monitor)

    elif monitor.OOD_approach == 'outlier':
        raise_alarm = is_pred_neg(yPred, intermediateValues, loaded_monitor)

    # just when GTSRB = ID and BTSC = OOD, otherwise comment the line below
    lbl = map_btsc_gtsrb(yPred)
    
    # OOD label numbers starts after the ID label numbers
    if lbl < classes_to_monitor: 
         
        if raise_alarm: 
            # it is not OOD but the monitor correctly detected a missclassification       
            if yPred != lbl: 
                monitor.arrTruePositive_ID[lbl].append(yPred) #True positives for missclassification in ID
                monitor.arr_pred_monitor_ID.append(1)
                monitor.arr_lbl_monitor_ID.append(1)
                
            # it is not OOD and the monitor wrongly detected a missclassification
            if yPred == lbl: 
                monitor.arrFalsePositive_ID[lbl].append(yPred) #False positives for missclassification in ID
                monitor.arr_pred_monitor_ID.append(1)
                monitor.arr_lbl_monitor_ID.append(0)
        else:
            # it is not OOD and the monitor wrongly missed a missclassification
            if yPred != lbl:
                monitor.arrFalseNegative_ID[lbl].append(yPred) #False negative for missclassification in ID
                monitor.arr_pred_monitor_ID.append(0)
                monitor.arr_lbl_monitor_ID.append(1) 
                        
            # it is not OOD and not a missclassification and the monitor correctly did nothing
            if yPred == lbl: 
                monitor.arrTrueNegative_ID[lbl].append(yPred) #True negatives for missclassification in ID
                monitor.arr_pred_monitor_ID.append(0)
                monitor.arr_lbl_monitor_ID.append(0)
    else:
        
        if raise_alarm: # it is OOD and the monitor correctly detected it
            monitor.arrTruePositive_OOD[lbl].append(yPred) #True positives for OOD
            monitor.arr_pred_monitor_OOD.append(1)
            monitor.arr_lbl_monitor_OOD.append(1)
           
        else: # it is OOD and the monitor wrongly detected it
            monitor.arrFalseNegative_OOD[lbl].append(yPred) #False negatives for OOD
            monitor.arr_pred_monitor_OOD.append(0)
            monitor.arr_lbl_monitor_OOD.append(1)

    return monitor


### False positive and true negative rates for OOD data are not comptabilized here.
### Since we are working with novelty class detection and all classes contained in the OOD dataset are unknown to the classifier. That is, all data is considered positive. 
### Therefore, the only possible monitors outcome are true positive (monitor detects it is OOD and it is OOD indeed) or false negative (monitor do not detect it is OOD despite it is OOD).  

## true negative rate ID at 95% of true positive rate OOD = verifies how much the monitor avoided false alarms in ID data when it reached 95% of the achieved detection on OOD data.
# It helps to understand if the monitor is capable of not interfering the performance of the classifier in known classes when trying to detect unknown classes. 

## false positive rate ID at 95% of true positive rate OOD = verifies how much the monitor raised false alarms in ID data when it reached 95% of the achieved detection on OOD data.
# It helps to understand if the monitor hinders the performance of the classifier in known classes when trying to detect unknown classes.

## 