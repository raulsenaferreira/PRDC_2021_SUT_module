
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


def safety_monitor_decision(readout, monitor, yPred, lbl, classes_to_monitor, intermediateValues,
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
    lbl = map_btsc_gtsrb(yPred, lbl)
    
    # OOD label numbers starts after the ID label numbers
    if lbl < classes_to_monitor: 
         
        if raise_alarm: 
            # wrong detection for OOD and correctly detection for missclassification       
            if yPred != lbl: 
                # correctly detected a missclassification
                try:
                    readout.arr_true_positive_ID[lbl].append(yPred) # True positives for missclassification in ID
                except:
                    readout.arr_true_positive_ID.update({lbl: [yPred]})

                readout.arr_pos_neg_ID_pred.append(1)
                readout.arr_pos_neg_ID_true.append(1)

                # incorrectly detected an ood
                try:
                    readout.arr_false_positive_OOD[lbl].append(yPred) # False positives for detection in OOD
                except:
                    readout.arr_false_positive_OOD.update({lbl: [yPred]})
                
                readout.arr_pos_neg_OOD_pred.append(1)
                readout.arr_pos_neg_OOD_true.append(0)
                
            # wrong detection for OOD and hindered a correct classification
            if yPred == lbl: 
                try:
                    readout.arr_false_positive_ID[lbl].append(yPred) #False positives for missclassification in ID
                except:
                    readout.arr_false_positive_ID.update({lbl: [yPred]})

                readout.arr_pos_neg_ID_pred.append(1)
                readout.arr_pos_neg_ID_true.append(0)
        else:
            # it is not OOD and the monitor missed a missclassification
            if yPred != lbl:
                try:
                    readout.arr_false_negative_ID[lbl].append(yPred) #False negative for missclassification in ID
                except:
                    readout.arr_false_negative_ID.update({lbl: [yPred]})

                readout.arr_pos_neg_ID_pred.append(0)
                readout.arr_pos_neg_ID_true.append(1) 
                        
            # it is not OOD and not a missclassification and the monitor correctly did nothing
            if yPred == lbl: 
                try:
                    readout.arr_true_negative_ID[lbl].append(yPred) #True negatives for missclassification in ID
                except:
                    readout.arr_true_negative_ID.update({lbl: [yPred]})

                readout.arr_pos_neg_ID_pred.append(0)
                readout.arr_pos_neg_ID_true.append(0)

                # OOD
                try:
                    readout.arr_true_negative_OOD[lbl].append(yPred) # True negative for detection in OOD
                except:
                    readout.arr_true_negative_OOD.update({lbl: [yPred]})

                readout.arr_pos_neg_OOD_pred.append(0)
                readout.arr_pos_neg_OOD_true.append(0)
    else:
        # it is OOD and the monitor correctly detected it
        if raise_alarm:
            try: 
                readout.arr_true_positive_OOD[lbl].append(yPred) #True positives for OOD
            except:
                readout.arr_true_positive_OOD.update({lbl: [yPred]})

            readout.arr_pos_neg_OOD_pred.append(1)
            readout.arr_pos_neg_OOD_true.append(1)

        # it is OOD and the monitor wrongly detected it
        else: 
            try:
                readout.arr_false_negative_OOD[lbl].append(yPred) #False negatives for OOD
            except:
                readout.arr_false_negative_OOD.update({lbl: [yPred]})

            readout.arr_pos_neg_OOD_pred.append(0)
            readout.arr_pos_neg_OOD_true.append(1)

    return readout


## true negative rate ID at 95% of true positive rate OOD = verifies how much the monitor avoided false alarms in ID data when it reached 95% of the achieved detection on OOD data.
# It helps to understand if the monitor is capable of not interfering the performance of the classifier in known classes when trying to detect unknown classes. 

## false positive rate ID at 95% of true positive rate OOD = verifies how much the monitor raised false alarms in ID data when it reached 95% of the achieved detection on OOD data.
# It helps to understand if the monitor hinders the performance of the classifier in known classes when trying to detect unknown classes.

## Correctly says that is in distribution while avoiding to say that the second is in distribution 