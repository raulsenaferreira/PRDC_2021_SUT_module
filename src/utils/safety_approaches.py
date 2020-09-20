
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


    if lbl < classes_to_monitor: # OOD label numbers starts after the ID label numbers
         
        if is_ood:        
            if yPred != lbl: 
                monitor.arrTruePositive_ID[yPred].append(lbl) #True positives
                
            if yPred == lbl: 
                monitor.arrFalsePositive_ID[yPred].append(lbl) #False positives

        else:
            if yPred != lbl:
                monitor.arrFalseNegative_ID[yPred].append(lbl) #False negative 
                        
            if yPred == lbl: 
                monitor.arrTrueNegative_ID[yPred].append(lbl) #True negatives
                
    else:

        if is_ood:
            monitor.arrTruePositive_OOD[yPred].append(lbl) #True positives
            
        else:
            monitor.arrFalseNegative_OOD[yPred].append(lbl) #False negative

    return monitor