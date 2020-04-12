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
import utils
import cv2
from PIL import Image
import scipy
import scipy.misc
import imageio
from skimage import data, img_as_float
from skimage import exposure
from skimage.filters import gaussian
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

                    
def make_abstraction(data, clusters, classe):
    data = np.asarray(data)

    #doing a projection by taking just the first and the last dimension of data
    data = data[:,[0,-1]]
    print(data.shape)
    dataByCluster={}
    for c, d in zip(clusters, data):
        #print(d)
        try:
            dataByCluster[c].append(d)
        except:
            dataByCluster.update({c:[d]})
    #print(np.asarray(dataByCluster[0]))

    array_box_by_cluster = {}

    array_box_by_cluster.update({classe:[]})

    for k, v in dataByCluster.items():
        arr_intermediate = []
        v = np.asarray(v)
        for i in range(v.shape[1]):
            min_i = np.amin(v[:,i])
            max_i = np.amax(v[:,i])
            arr_intermediate.append([min_i, max_i])
        array_box_by_cluster[classe].append(arr_intermediate)

    return array_box_by_cluster


def get_activ_func(model, image, layerName):
    inter_output_model = Model(inputs = model.input, outputs = model.get_layer(layerName).output) #last layer: index 7 or name 'dense'
    return inter_output_model.predict(image)


def contrast_normalization(image):
    X = np.array(image)
    
    image_blur = cv2.GaussianBlur(image,(65,65),10)
    # new_image = cv2.subtract(img,image_blur).astype('float32') # WRONG, the result is not stored in float32 directly
    new_image = cv2.subtract(image,image_blur, dtype=cv2.CV_32F)
    out = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #res = np.hstack((X, out)) #stacking images side-by-side
    #imageio.imwrite('Contrast.jpg', res)
    return out


def image_adjustment(image):
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))

    #res = np.hstack((image, img_rescale)) #stacking images side-by-side
    #imageio.imwrite('Imadjust.jpg', res)
    return img_rescale


def histogram_equalization(img):
    equ =  exposure.equalize_hist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    
    #imageio.imwrite('Histeq.jpg',res)
    return equ


def adaptive_hist_eq(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.3)

    #res = np.hstack((img,img_adapteq)) #stacking images side-by-side
    #imageio.imwrite('Adapthisteq.jpg',res)
    return img_adapteq


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
height = 30
width = 30
channels = 3
num_classes = 43
n_inputs = height * width*channels
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
model_name = 'GTS_model_DNN_ensemble_0.h5'
X_train = np.array(list(map(image_adjustment, X_train)))
X_valid = np.array(list(map(image_adjustment, X_valid)))

model_name = 'GTS_model_DNN_ensemble_1.h5'
X_train = np.array(list(map(histogram_equalization, X_train)))
X_valid = np.array(list(map(histogram_equalization, X_valid)))

model_name = 'GTS_model_DNN_ensemble_2.h5'
X_train = np.array(list(map(adaptive_hist_eq, X_train)))
X_valid = np.array(list(map(adaptive_hist_eq, X_valid)))

model_name = 'GTS_model_DNN_ensemble_3.h5'
X_train = np.array(list(map(contrast_normalization, X_train)))
X_valid = np.array(list(map(contrast_normalization, X_valid)))


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