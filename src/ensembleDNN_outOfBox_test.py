import sys
import os
import numpy as np
import cv2
import dataset
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import confusion_matrix_pretty_print as cmpp
import pickle
from sklearn.metrics import confusion_matrix
from PIL import Image
import tensorflow as tf
from tensorflow import keras


def get_activ_func(model, image, layerName):
    inter_output_model = Model(inputs = model.input, outputs = model.get_layer(layerName).output) #last layer: index 7 or name 'dense'
    
    return inter_output_model.predict(image)



number_of_DNNs = 5

def find_point_box_ensemble(intermediateValues_all, class_to_monitor):
    result = False
    for i in range(number_of_DNNs):
        box = pickle.load(open("box_ensemble_results"+sep+"runtime_monitor_Box_DNN_"+str(i)+".p", "rb"))
        data = np.asarray(intermediateValues_all[i])
        #print(intermediateValues_all)
        x = data[0]
        y = data[-1]
        #print("point:", x, y)
        
        try:
            for box in boxes[class_to_monitor]:
                #B = box[0]
                #print(box)
                x1 = box[0][0]
                x2 = box[0][1]
                y1 = box[1][0]
                y2 = box[1][1]
                if x >= x1 and x <= x2 and y >= y1 and y <= y2: 
                    return True
                else : 
                    result = False
        except:
            result = False
    return result


is_windows = sys.platform.startswith('win')
sep = '\\'
    
if is_windows == False:
    sep = '/'


classToMonitor = 7
count = 0
count2 = 0
arrPred = []
arrLabel = []

arrFalseNegative = {str(classToMonitor): 0}
arrTrueNegative = {str(classToMonitor): 0}
arrFalsePositive = {str(classToMonitor): 0}
arrTruePositive = {str(classToMonitor): 0}

isTestOneClass = True
height = 30
width = 30
channels = 3
num_classes = 43
n_inputs = height * width*channels

testPath = os.getcwd()+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep

y_test=pd.read_csv(testPath+"Test.csv")
#y_test=pd.read_csv(testPath+"Train.csv")
#print(y_test.head())
labels=y_test['Path'].values
y_test=y_test['ClassId'].values
y_test=np.array(y_test)

d=[]
for j in labels:
    i1=cv2.imread(testPath+j)
    i2=Image.fromarray(i1,'RGB')
    i3=i2.resize((height,width))
    d.append(np.array(i3))

X_test=np.array(d)
X_test = X_test.astype('float32')/255 
#y_train = to_categorical(y_train, num_classes) # Using one hote encoding

print("Test :", X_test.shape, y_test.shape)

# Recreate the exact same model purely from the file
model_0 = load_model('GTS_model_DNN_ensemble_0.h5')
model_1 = load_model('GTS_model_DNN_ensemble_1.h5')
model_2 = load_model('GTS_model_DNN_ensemble_2.h5')
model_3 = load_model('GTS_model_DNN_ensemble_3.h5')
model_4 = load_model('GTS_model_DNN_ensemble_4.h5')

counter = 0
loading_percentage = 0.1
loaded = int(loading_percentage*len(y_test))
print(loaded)

for img, lab in zip(X_test, y_test):
    counter+=1
    if counter % loaded == 0:
        print("{} % progress...".format(int(loading_percentage*100)))
        loading_percentage+=0.1
        
    img = np.asarray([img])
    y_0 = model_0.predict(img)
    y_1 = model_1.predict(img)
    y_2 = model_2.predict(img)
    y_3 = model_3.predict(img)
    y_4 = model_4.predict(img)
    y_all = np.vstack((y_0[0],y_1[0],y_2[0],y_3[0],y_4[0]))
    
    y_all = np.average(y_all, axis=0)
    #print(y_all)
    #break
    yPred = np.argmax(y_all)
    #print('yPred: ',yPred)
    arrPred.append(yPred)

    intermediateValues_0 = get_activ_func(model_0, img, 'dense_1')[0]
    intermediateValues_1 = get_activ_func(model_1, img, 'dense_1')[0]
    intermediateValues_2 = get_activ_func(model_2, img, 'dense_1')[0]
    intermediateValues_3 = get_activ_func(model_3, img, 'dense_1')[0]
    intermediateValues_4 = get_activ_func(model_4, img, 'dense_1')[0]
    intermediateValues_all = np.vstack((intermediateValues_0, intermediateValues_1, intermediateValues_2, intermediateValues_3, intermediateValues_4))

    # ML is incorrect but monitor does not trust on it = TP
    # ML is correct but monitor does not trust on it = FP
    # ML is incorrect and monitor trust on it = FN
    # ML is correct and monitor trust on it = TN

    if yPred == classToMonitor:
        if find_point_box_ensemble(intermediateValues_all, yPred):
            count += 1
            if yPred != lab:
                arrFalseNegative[str(classToMonitor)] += 1 #False negative			
            if yPred == lab: 
                arrTrueNegative[str(classToMonitor)] += 1 #True negatives
        else:
            count2 += 1
            if yPred != lab: 
                arrTruePositive[str(classToMonitor)] += 1 #True positives
            if yPred == lab: 
                arrFalsePositive[str(classToMonitor)] += 1 #False positives
    elif yPred != lab and lab == classToMonitor:
        print("missclassification of class: {0}({1}). Predicted class: {2}({3})".format(str(lab), y_all[lab], yPred, y_all[yPred]))

print("Similar patterns (FN + TN): ", count)
print("Raised alarms (FP + TP): ", count2)

if isTestOneClass:
    print("FP: {}={}, Total={}".format(classToMonitor, arrFalsePositive[str(classToMonitor)], sum(arrFalsePositive.values()))) 
    print("FN: {}={}, Total={}".format(classToMonitor, arrFalseNegative[str(classToMonitor)], sum(arrFalseNegative.values()))) 
    print("TP: {}={}, Total={}".format(classToMonitor, arrTruePositive[str(classToMonitor)], sum(arrTruePositive.values())))
    print("TN: {}={}, Total={}".format(classToMonitor, arrTrueNegative[str(classToMonitor)], sum(arrTrueNegative.values())))

#CM = confusion_matrix(arrLabel, arrPred)
#print("confusion matrix", CM)

a = pd.crosstab(pd.Series(y_test), pd.Series(arrPred), rownames=['True'], colnames=['Predicted'], margins=True)
print("confusion matrix - no monitor", a)

#incrementing alpha
#monitor.enlargeSetByOneBitFluctuation(stopSignClass)
