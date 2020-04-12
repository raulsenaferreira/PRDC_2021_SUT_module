import sys
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from utils import util
from utils import abstraction_box
from tensorflow import keras
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix


is_windows = sys.platform.startswith('win')
sep = '\\'
    
if is_windows == False:
    sep = '/'

classToMonitor = 7
count = [0, 0]
arrPred = []
num_classes = 43
isTestOneClass = True
layer_name = 'dense_1'

models_folder = "bin"+sep+"models"+sep
monitors_folder = "bin"+sep+"monitors"+sep
script_path = os.getcwd()
testPath = str(Path(script_path).parent)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep
#loading test set
X_test, y_test = util.load_GTRSB_csv(testPath, "Test.csv")

#3 variables for log (optional)
counter = 0
loading_percentage = 0.1
loaded = int(loading_percentage*len(y_test))

# loading model and abstraction boxes
model = load_model(models_folder+'CNN_model_GTSRB.h5')
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
    intermediateValues = util.get_activ_func(model, img, layer_name)[0]

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
        elif lab==classToMonitor:
            print("missclassification --- new pattern for class",yPred, str(lab))

print("Similar patterns (FN + TN): ", count[0])
print("Raised alarms (FP + TP): ", count[1])
util.print_positives_negatives(arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative, classToMonitor, isTestOneClass)
a = pd.crosstab(pd.Series(y_test), pd.Series(arrPred), rownames=['True'], colnames=['Predicted'], margins=True)
print("confusion matrix - no monitor", a)