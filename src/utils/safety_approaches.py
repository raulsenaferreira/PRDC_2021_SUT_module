
def is_pred_diff(yPred, intermediateValues, loaded_monitor):
    
    yPred_by_monitor = loaded_monitor.predict(intermediateValues)
    #print(np.shape(yPred_by_monitor))

    if yPred_by_monitor == yPred:
        return False
    
    return True


def safety_monitor_decision(monitor, yPred, lbl, classes_to_monitor, intermediateValues,
 scaler, loaded_monitor):
    
    # if you want to scale act func values
    if scaler != None:
        intermediateValues = scaler.transform(intermediateValues)

    is_ood = False

    if monitor.OOD_approach == 'equality':
        # if monitor acceptance approach is based on two equal predictions
        is_ood = is_pred_diff(yPred, intermediateValues, loaded_monitor)

    elif monitor.OOD_approach == '':
        pass

    # OOD label numbers starts after the ID label numbers
    if lbl < classes_to_monitor: 
         
        if is_ood:        
            if yPred != lbl: 
                monitor.arrTruePositive_ID[lbl].append(yPred) #True positives
                monitor.arr_pred_monitor_ID.append(1)
                monitor.arr_lbl_monitor_ID.append(1)
                
            if yPred == lbl: 
                monitor.arrFalsePositive_ID[lbl].append(yPred) #False positives
                monitor.arr_pred_monitor_ID.append(1)
                monitor.arr_lbl_monitor_ID.append(0)
        else:
            if yPred != lbl:
                monitor.arrFalseNegative_ID[lbl].append(yPred) #False negative
                monitor.arr_pred_monitor_ID.append(0)
                monitor.arr_lbl_monitor_ID.append(1) 
                        
            if yPred == lbl: 
                monitor.arrTrueNegative_ID[lbl].append(yPred) #True negatives
                monitor.arr_pred_monitor_ID.append(0)
                monitor.arr_lbl_monitor_ID.append(0)
    else:

        if is_ood:
            monitor.arrTruePositive_OOD[lbl].append(yPred) #True positives
            monitor.arr_pred_monitor_OOD.append(1)
            monitor.arr_lbl_monitor_OOD.append(1)
            
        else:
            monitor.arrFalseNegative_OOD[lbl].append(yPred) #False negative
            monitor.arr_pred_monitor_OOD.append(0)
            monitor.arr_lbl_monitor_OOD.append(1)

    return monitor