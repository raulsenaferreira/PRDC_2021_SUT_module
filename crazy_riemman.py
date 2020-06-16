import sys
import os
from pathos.multiprocessing import ProcessingPool as Pool
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import gzip
from PIL import Image
import scipy.io as spio
from skimage.transform import resize
import cv2
from sklearn.model_selection import train_test_split
from src.utils import plot_functions as pf


'''
The context: https://www.quora.com/What-makes-it-so-difficult-to-prove-or-disprove-the-Riemann-hypothesis
Some interesting questions:
1) Who is better on read rotating images? AI or Humans?
2) How to make a public experimentation and collect the results?
3) How to take random stories from a set of non-known novels and display raotating text for this?
4) How to make the same in 3) and give to the CNN classify it ?
5) What are the speeds that produced good endings? 
6) The speeds found in 5) are the same between humans? are the same between humans and CNN?
7) These speeds cannot be predicted? (the theory says no, they dont)

TODO:

DONE:
- 
'''



if __name__ == "__main__":
	def get_separator():
		is_windows = sys.platform.startswith('win')
		sep = '\\'

		if is_windows == False:
			sep = '/'

		return sep

	sep = get_separator()
	
	def decoding_data(images, labels, num, dim):
		#emnist_map = {}
		data = np.zeros(num * dim, dtype=np.uint8).reshape((num, dim))
		target = np.zeros(num, dtype=np.uint8).reshape((num, ))

		with gzip.open(images, 'rb') as f_images, gzip.open(labels, 'rb') as f_labels:
			f_images.read(16)
			f_labels.read(8)
			for i in range(num):
				target[i] = ord(f_labels.read(1))
				#emnist_map[target[i]]=f_labels.read(1)
				for j in range(dim):
					data[i, j] = ord(f_images.read(1))

		return data, target 


	def emnist_preprocess(images):
		modified_images = []
		for image in images:
			image = np.fliplr(image)
			modified_images.append(np.rot90(image))
			#image = cv2.flip(image, 1)
			#modified_images.append(np.rot90(image, k=1, axes=(0,1)))
		return np.asarray(modified_images)


	def load_balanced_emnist():
		train_images = 'data'+sep+'ConceptMNIST'+sep+'e_mnist'+sep+'emnist-balanced-train-images-idx3-ubyte.gz'
		train_labels = 'data'+sep+'ConceptMNIST'+sep+'e_mnist'+sep+'emnist-balanced-train-labels-idx1-ubyte.gz'
		test_images = 'data'+sep+'ConceptMNIST'+sep+'e_mnist'+sep+'emnist-balanced-test-images-idx3-ubyte.gz'
		test_labels = 'data'+sep+'ConceptMNIST'+sep+'e_mnist'+sep+'emnist-balanced-test-labels-idx1-ubyte.gz'
		num_train = 112800
		num_test = 18800
		dim = 28
		data_train, target_train = decoding_data(train_images, train_labels, num_train, dim*dim)
		data_test, target_test = decoding_data(test_images, test_labels, num_test, dim*dim)
		data_train = data_train.reshape(num_train, dim, dim, 1)
		data_test = data_test.reshape(num_test, dim, dim, 1)
		
		return (data_train, target_train), (data_test, target_test)


	def text_to_nist(text):
		return [ord(t)-97 for t in text]

	def nist_to_text(array):
		return ''.join([chr(num+97) for num in array])


	def emnist_upper_lower_letters_mapping(array):
		#mapping according the structure of the dataset https://arxiv.org/abs/1702.05373v1
		for i in range(len(array)):
			if array[i]>35 and array[i]<38:
				array[i] -= 26
			elif array[i]>37 and array[i]<43:
				array[i]-=25
			elif array[i]==43:
				array[i]-=20
			elif array[i]>43 and array[i]<46:
				array[i]-=18
			elif array[i]==46:
				array[i]-=17

		return array


	def model_building(x_train, y_train):
		num_classes = 47
		# Normalise
		x_train = x_train.astype('float32')
		x_train /= 255

		# One hot encoding
		y_train = np_utils.to_categorical(y_train, num_classes)
		
		# partition to train and val
		x_train, val_x, y_train, val_y = train_test_split(x_train, y_train, test_size= 0.10, random_state=7)

		model = Sequential()

		model.add(Conv2D(filters=128, kernel_size=(5,5), padding = 'same', activation='relu',\
						 input_shape=(28, 28, 1)))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
		model.add(Conv2D(filters=64, kernel_size=(3,3) , padding = 'same', activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Flatten())
		model.add(Dense(units=128, activation='relu'))
		model.add(Dropout(.5))
		model.add(Dense(units=num_classes, activation='softmax'))

		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(x_train, y_train, epochs=10, batch_size=512, verbose=1, validation_data=(val_x, val_y))

		#saving model
		model.save('emnist_cnn.h5')



	#test conversion
	#text = 'abcdefghijklmnopqrstuvwxyz'
	#array = text_to_nist(text)
	#print(nist_to_text(array))
	
	#emnist data
	(x_train, y_train), (x_test, y_test) = load_balanced_emnist()
	#correcting emnist image orientation
	x_test = emnist_preprocess(x_test)

	#how is mapped numbers and labels?
	#unique_classes, indices = np.unique(y_test, return_index=True)
	#unique_images = x_test[indices]
	#plot some examples
	#pf.plot_images(unique_classes, unique_images, 5, 9)

	#associating lower and upper case letters
	y_test = emnist_upper_lower_letters_mapping(y_test)
	#plot some examples
	#pf.plot_images(x_test, y_test, 5, 9)

	#TODO:
	#emnist letters come after digits
	#array +=10
	
	#training and saving the model
	#model_building(x_train, y_train)

	#classifying
	x_test = x_test.astype('float32')
	x_test /= 255
	y_test = np_utils.to_categorical(y_test, 47)
	model = load_model('emnist_cnn.h5')
	#How good the model is performing on test data?
	score = model.evaluate(test_x, test_y, verbose=0)
	print("Test loss:", score[0])
	print("Test accuracy:", score[1])

	#confusion matrix 
	y_pred = model.predict(x_test)
	y_pred = (y_pred > 0.5)
	cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))