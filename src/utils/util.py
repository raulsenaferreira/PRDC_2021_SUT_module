from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
import cv2
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from PIL import Image
import scipy
import scipy.misc
import imageio
from skimage import data, img_as_float
from skimage import exposure
from skimage.filters import gaussian


is_windows = sys.platform.startswith('win')
sep = '\\'

if is_windows == False:
	sep = '/'


def get_activ_func(model, image, layerName=None, layerIndex=None):
	inter_output_model = Model(inputs = model.input, outputs = model.get_layer(name=layerName, index=layerIndex).output) #last layer: index 7 or name 'dense'
	return inter_output_model.predict(image)


def loading_info(counter, loaded, loading_percentage):
	counter+=1
	if counter % loaded == 0:
		print("{} % loaded".format(int(loading_percentage*100)))
		loading_percentage+=0.1
	return counter, loading_percentage


def load_GTRSB_dataset(trainPath, val_size):
	# Reading the input images and putting them into a numpy array
	data=[]
	labels=[]
	height = 30
	width = 30
	channels = 3
	num_classes = 43
	n_inputs = height * width*channels

	for i in range(num_classes) :
		path = trainPath+str(i)+sep
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

	s=np.arange(x_train.shape[0])
	np.random.seed(num_classes)
	np.random.shuffle(s)

	x_train=x_train[s]
	y_train=y_train[s]
	# Split Data
	X_train,X_valid,Y_train,Y_valid = train_test_split(x_train,y_train,test_size = val_size,random_state=0)
	#Using one hote encoding for the train and validation labels
	Y_train = to_categorical(Y_train, num_classes)
	Y_valid = to_categorical(Y_valid, num_classes)
	print("Training set shape :", X_train.shape)
	print("Validation set shape :", X_valid.shape)
	
	return X_train,X_valid,Y_train,Y_valid


def load_GTRSB_csv(testPath, filename):
	height = 30
	width = 30
	channels = 3
	n_inputs = height * width*channels
	y_test=pd.read_csv(testPath+filename)
	labels=y_test['Path'].values
	y_test=y_test['ClassId'].values
	y_test=np.array(y_test)

	d=[]
	for j in labels:
		#print("a {} b {}".format(testPath, j))
		i1=cv2.imread(testPath+j)
		i2=Image.fromarray(i1,'RGB')
		i3=i2.resize((height,width))
		d.append(np.array(i3))

	X_test=np.array(d)
	X_test = X_test.astype('float32')/255 
	print("Test :", X_test.shape, y_test.shape)
	return X_test, y_test


def print_positives_negatives(arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative, classToMonitor, isTestOneClass=True):
	if isTestOneClass:
		print("FP: {}={}, Total={}".format(classToMonitor, arrFalsePositive[str(classToMonitor)], sum(arrFalsePositive.values()))) 
		print("FN: {}={}, Total={}".format(classToMonitor, arrFalseNegative[str(classToMonitor)], sum(arrFalseNegative.values()))) 
		print("TP: {}={}, Total={}".format(classToMonitor, arrTruePositive[str(classToMonitor)], sum(arrTruePositive.values())))
		print("TN: {}={}, Total={}".format(classToMonitor, arrTrueNegative[str(classToMonitor)], sum(arrTrueNegative.values())))

#image pre-processing methods
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