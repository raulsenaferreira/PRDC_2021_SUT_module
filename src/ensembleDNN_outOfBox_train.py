from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
import matplotlib.pyplot as plt
import dataset
import time
from datetime import timedelta
import math
import random
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
from utils import util
from utils import abstraction_box
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


is_windows = sys.platform.startswith('win')
sep = '\\'

if is_windows == False:
    sep = '/'

# Reading the input images and putting them into a numpy array
sizeOfNeuronsToMonitor = 256
filteringRate = 0.3 #same from the neuron pattern paper. Put 0 if you want to monitor all neurons
classToMonitor = 7
data=[]
labels=[]
arrWeights = []
arrPred = []
arrLabel = []
trainPath = os.getcwd()+sep+'data'+sep+'GTS_dataset'+sep

'''
for i in range(num_classes) :
    path = trainPath+"kaggle"+sep+"Train"+sep+str(i)+sep
    #print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")
  
x_train=np.array(data)
x_train= x_train/255.0

y_train=np.array(labels)
#y_train = to_categorical(y_train, num_classes) # Using one hote encoding

s=np.arange(x_train.shape[0])
np.random.seed(43)
np.random.shuffle(s)
x_train=x_train[s]
y_train=y_train[s]

# Split Data
X_train,X_valid,Y_train,Y_valid = train_test_split(x_train,y_train,test_size = 0.3,random_state=0)
print("Train :", X_train.shape)
print("Valid :", X_valid.shape)
'''
#pre-processing images procedures
#image_adjustment(X_train[0])
#histogram_equalization(X_train[0])
#adaptive_hist_eq(X_train[0])
#contrast_normalization(X_train[0])
'''



#model with original data
model_name = 'GTS_model_DNN_ensemble_4.h5'


#Using one hote encoding for the train and validation labels
Y_train = to_categorical(Y_train, 43)
Y_valid = to_categorical(Y_valid, 43)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)
epochs = 10
history = model.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_data=(X_valid, Y_valid))
model.save(model_name)
'''

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


print("Test :", X_test.shape, y_test.shape)

model_0 = load_model('GTS_model_DNN_ensemble_0.h5')
model_1 = load_model('GTS_model_DNN_ensemble_1.h5')
model_2 = load_model('GTS_model_DNN_ensemble_2.h5')
model_3 = load_model('GTS_model_DNN_ensemble_3.h5')
model_4 = load_model('GTS_model_DNN_ensemble_4.h5')
arrWeights_0 = []
arrWeights_1 = []
arrWeights_2 = []
arrWeights_3 = []
arrWeights_4 = []
arrPred = []
K = 3
for img, lab in zip(X_test, y_test): 
    img = np.asarray([img])
    y_0 = model_0.predict(img)
    y_1 = model_1.predict(img)
    y_2 = model_2.predict(img)
    y_3 = model_3.predict(img)
    y_4 = model_4.predict(img)
    y_all = np.vstack((y_0[0],y_1[0],y_2[0],y_3[0],y_4[0]))
    #print(y_all)
    y_all = np.average(y_all, axis=0)
    #print(y_all)
    #break
    yPred = np.argmax(y_all)
    #print('yPred: ',yPred)
    arrPred.append(yPred)

    if yPred == lab and yPred==classToMonitor:
        #print("ok")
        arrWeights_0.append(get_activ_func(model_0, img, 'dense_1')[0])
        arrWeights_1.append(get_activ_func(model_1, img, 'dense_1')[0])
        arrWeights_2.append(get_activ_func(model_2, img, 'dense_1')[0])
        arrWeights_3.append(get_activ_func(model_3, img, 'dense_1')[0])
        arrWeights_4.append(get_activ_func(model_4, img, 'dense_1')[0])


clusters_0 = KMeans(n_clusters=K, random_state=0).fit_predict(arrWeights_0)
clusters_1 = KMeans(n_clusters=K, random_state=0).fit_predict(arrWeights_1)
clusters_2 = KMeans(n_clusters=K, random_state=0).fit_predict(arrWeights_2)
clusters_3 = KMeans(n_clusters=K, random_state=0).fit_predict(arrWeights_3)
clusters_4 = KMeans(n_clusters=K, random_state=0).fit_predict(arrWeights_4)

print("Accuracy Ensemble:", accuracy_score(y_test, arrPred))
print("making boxes...")
boxes_0 = make_abstraction(arrWeights_0, clusters_0, classToMonitor)
boxes_1 = make_abstraction(arrWeights_1, clusters_1, classToMonitor)
boxes_2 = make_abstraction(arrWeights_2, clusters_2, classToMonitor)
boxes_3 = make_abstraction(arrWeights_3, clusters_3, classToMonitor)
boxes_4 = make_abstraction(arrWeights_4, clusters_4, classToMonitor)

print("Saving boxes in a file")
pickle.dump(boxes_0, open( "box_ensemble_results"+sep+"runtime_monitor_Box_DNN_0.p", "wb" ))
pickle.dump(boxes_1, open( "box_ensemble_results"+sep+"runtime_monitor_Box_DNN_1.p", "wb" ))
pickle.dump(boxes_2, open( "box_ensemble_results"+sep+"runtime_monitor_Box_DNN_2.p", "wb" ))
pickle.dump(boxes_3, open( "box_ensemble_results"+sep+"runtime_monitor_Box_DNN_3.p", "wb" ))
pickle.dump(boxes_4, open( "box_ensemble_results"+sep+"runtime_monitor_Box_DNN_4.p", "wb" ))

'''
y_test = to_categorical(y_test, num_classes) # Using one hote encoding
score, acc = model_0.evaluate(X_test, y_test)
print("Accuracy model 0:", acc)
score, acc = model_1.evaluate(X_test, y_test)
print("Accuracy model 1:", acc)
score, acc = model_2.evaluate(X_test, y_test)
print("Accuracy model 2:", acc)
score, acc = model_3.evaluate(X_test, y_test)
print("Accuracy model 3:", acc)
score, acc = model_4.evaluate(X_test, y_test)
print("Accuracy model 4:", acc)
'''