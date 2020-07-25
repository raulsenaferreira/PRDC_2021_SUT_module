from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
import cv2
import random
import numpy as np
from keras.models import Model
import scipy
import scipy.misc
import imageio
from skimage import data, img_as_float
from skimage import exposure
from skimage.filters import gaussian
from keras.preprocessing.image import ImageDataGenerator


def get_separator():
	is_windows = sys.platform.startswith('win')
	sep = '\\'

	if is_windows == False:
		sep = '/'

	return sep


def is_monitored_prediction(yPred, label, classToMonitor):
	return yPred == label and yPred==classToMonitor


def get_activ_func(model, image, layerIndex):
	inter_output_model = Model(inputs = model.input, outputs = model.get_layer(index=layerIndex).output) #last layer: index 7 or name 'dense'
	return inter_output_model.predict(image)


def loading_info(counter, loaded, loading_percentage):
	counter+=1
	if counter % loaded == 0:
		print("{} % processed".format(int(loading_percentage*100)))
		loading_percentage+=0.1
	return counter, loading_percentage


def print_positives_negatives(arrFalsePositive, arrFalseNegative, arrTruePositive, arrTrueNegative, classToMonitor, isTestOneClass=True):
	if isTestOneClass:
		print("Monitored class:", classToMonitor)
		print("FP: {}".format(arrFalsePositive[str(classToMonitor)]))
		print("FN: {}".format(arrFalseNegative[str(classToMonitor)])) 
		print("TP: {}".format(arrTruePositive[str(classToMonitor)]))
		print("TN: {}".format(arrTrueNegative[str(classToMonitor)]))
		print("Similar patterns (FN + TN): ", sum(arrFalseNegative.values()) + sum(arrTrueNegative.values()))
		print("Raised alarms (FP + TP): ", sum(arrFalsePositive.values()) + sum(arrTruePositive.values()))


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


def rescale(trainX, trainY, model, epochs, batch_size):
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	train_iterator = datagen.flow(trainX, trainY, batch_size=batch_size)
	return train_iterator


def std_normalization(trainX, trainY, model, epochs, batch_size):
	datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
	datagen.fit(trainX)
	train_iterator = datagen.flow(trainX, trainY, batch_size=batch_size)
	return train_iterator