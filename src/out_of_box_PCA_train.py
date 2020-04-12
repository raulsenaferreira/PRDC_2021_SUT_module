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
from sklearn.cluster import KMeans
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
import utils
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
from sklearn.decomposition import PCA


is_windows = sys.platform.startswith('win')
sep = '\\'

if is_windows == False:
    sep = '/'


def get_activ_func(model, image, layerName):
    inter_output_model = Model(inputs = model.input, outputs = model.get_layer(layerName).output) #last layer: index 7 or name 'dense'
    return inter_output_model.predict(image)


def make_abstraction(data, clusters, classe):
	data = np.asarray(data)

	#doing a projection using PCA with 2 components
	#data = data[:,[0,-1]]
	pca = PCA(n_components=2)
	pca = pca.fit(data)
	pickle.dump(pca, open( "runtime_monitor_trained_PCA.p", "wb" ))
	data = pca.transform(data)
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


# Reading the input images and putting them into a numpy array
sizeOfNeuronsToMonitor = 256
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
model.save('GTS_model.h5')
'''

#Y_valid = Y_valid[:1000]
model = load_model('GTS_model.h5')
counter = 0
loading_percentage = 0.1
loaded = int(loading_percentage*len(Y_valid))
print(loaded)
print("Getting neuron patterns")
for img, lab in zip(X_valid, Y_valid):
    counter+=1
    if counter % loaded == 0:
        print("{} % loaded".format(int(loading_percentage*100)))
        loading_percentage+=0.1
        
    img = np.asarray([img])
    yPred = np.argmax(model.predict(img))
    if yPred == lab and yPred==classToMonitor:
        #print("ok")
        arrWeights.append(get_activ_func(model, img, 'dense_1')[0])
        #arrPred.append(yPred)
        #arrLabel.append(lab)


clusters = KMeans(n_clusters=3, random_state=0).fit_predict(arrWeights)
'''
print("processing abstract box\n Finding best K for clustering...")
#apply clustering
K = 1
kmeans = KMeans(n_clusters=K, random_state=0).fit(arrWeights)
inertia = kmeans.inertia_
threshold = 0.3 #reported threshold for the GTRSB dataset in the outside-of-box paper
while inertia > threshold:
	K+=1
	kmeans = KMeans(n_clusters=K, random_state=0).fit(arrWeights)
	inertia = kmeans.inertia_
	print("K and inertia:", K, inertia)

print("Clustering...")
clusters = kmeans.predict(arrWeights)
'''
print("making boxes...")
boxes = make_abstraction(arrWeights, clusters, classToMonitor)
print("Saving boxes in a file")
pickle.dump(boxes, open( "runtime_monitor_Box_PCA.p", "wb" ))


'''
#testing
v0 = [-2, 1, 0, 3, 1]
v1 = [0, 3, 1, -2, 2]
v2 = [1, 0, -2, -3, 3]
v3 = [-1, -1, 2, 0, -1]
v4 = [3, -3, -1, 2, -2]
v5 = [-3, -2, -3, 3, 0]

data = np.asarray([v0, v1, v2, v3, v4, v5])
print("actv function", data)
#print("2D projection", data[:,[0,-1]])

K = 1
kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
inertia = kmeans.inertia_
threshold = 50 #reported threshold for the GTRSB dataset in the out-of-box paper
while inertia > threshold:
	K+=1
	kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
	inertia = kmeans.inertia_
	print("K and inertia:", K, inertia)

array_box = []
for i in range(data.shape[1]):
	min_i = np.amin(data[:,i])
	max_i = np.amax(data[:,i])
	array_box.append([min_i, max_i])
print("activ func bounds:", array_box)

clusters = kmeans.predict(data)
print("clusters activ func:", clusters)

#doing a projection by taking just the first and the last dimension of data
data = data[:,[0,-1]]

dataByCluster={}
for c, d in zip(clusters, data):
	#print(d)
	try:
		dataByCluster[c].append(d)
	except:
		dataByCluster.update({c:[d]})
#print(np.asarray(dataByCluster[0]))



array_box_by_cluster = {}

classe = 14
array_box_by_cluster.update({classe:[]})

for k, v in dataByCluster.items():
	arr_intermediate = []
	v = np.asarray(v)
	for i in range(v.shape[1]):
		min_i = np.amin(v[:,i])
		max_i = np.amax(v[:,i])
		arr_intermediate.append([min_i, max_i])
	array_box_by_cluster[classe].append(arr_intermediate)
	
	print("bounds of clustered ({}) activ func: {}".format(k, array_box_by_cluster[classe]))


test = np.asarray([[-3, -2, -3, 3, 0], [-9, -8, -7, 9, 5]])
#after clustering ...
test = test[:,[0,-1]]
for X in test:
	if not find_point(array_box_by_cluster[classe], X[0], X[1]):
		print("outside of the box!", X)

'''