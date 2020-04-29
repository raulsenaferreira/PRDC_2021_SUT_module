from src.utils import util
import keras
from keras.datasets import mnist
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class Dataset:
	"""docstring for Dataset"""
	def __init__(self, dataset_name):
		super(Dataset, self).__init__()
		self.dataset_name = dataset_name
		self.width = 28
		self.height = 28
		self.channels = 0
		self.sep = util.get_separator()
		self.testPath = ''
		self.num_classes = 0
		self.trainPath = ''
		self.testPath = ''
		self.validation_size = None
	

	def load_mnist(self, onehotencoder=True):
		self.num_classes = 10
		# input image dimensions
		img_rows, img_cols, img_dim = 28, 28, 1

		# the data, split between train and test sets
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		if K.image_data_format() == 'channels_first':
			x_train = x_train.reshape(x_train.shape[0], img_dim, img_rows, img_cols)
			x_test = x_test.reshape(x_test.shape[0], img_dim, img_rows, img_cols)
			input_shape = (img_dim, img_rows, img_cols)
		else:
			x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_dim)
			x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_dim)
			input_shape = (img_rows, img_cols, img_dim)

		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255

		x_train, x_valid = train_test_split(x_train, test_size=self.validation_size, shuffle=False)
		y_train, y_valid = train_test_split(y_train, test_size=self.validation_size, shuffle=False)

		print('x_train shape:', x_train.shape)
		print(x_train.shape[0], 'train samples')
		print(x_test.shape[0], 'test samples')

		# convert class vectors to binary class matrices
		if onehotencoder:
			y_train = keras.utils.to_categorical(y_train, self.num_classes)
			y_valid = keras.utils.to_categorical(y_valid, self.num_classes)
			y_test = keras.utils.to_categorical(y_test, self.num_classes)

		return x_train, y_train, x_valid, y_valid, x_test, y_test, input_shape


	def load_GTRSB_dataset(self, onehotencoder=True):
		# Reading the input images and putting them into a numpy array
		sep = get_separator()
		data=[]
		labels=[]
		
		n_inputs = self.height * self.width * self.channels

		for i in range(self.num_classes) :
			path = self.trainPath+str(i)+self.sep
			#print(path)
			Class=os.listdir(path)
			for a in Class:
				try:
					image=cv2.imread(path+a)
					image_from_array = Image.fromarray(image, 'RGB')
					size_image = image_from_array.resize((self.height, self.width))
					data.append(np.array(size_image))
					labels.append(i)
				except AttributeError:
					print(" ")
		  
		x_train=np.array(data)
		x_train= x_train/255.0
		y_train=np.array(labels)

		s=np.arange(x_train.shape[0])
		np.random.seed(self.num_classes)
		np.random.shuffle(s)

		x_train=x_train[s]
		y_train=y_train[s]
		# Split Data
		X_train,X_valid,Y_train,Y_valid = train_test_split(x_train,y_train,test_size = self.validation_size,random_state=0)
		
		if onehotencoder:
			#Using one hote encoding for the train and validation labels
			Y_train = to_categorical(Y_train, self.num_classes)
			Y_valid = to_categorical(Y_valid, self.num_classes)
		print("Training set shape :", X_train.shape)
		print("Validation set shape :", X_valid.shape)
		
		return X_train,X_valid,Y_train,Y_valid


	def load_GTRSB_csv(self, filename):
		n_inputs = self.height * self.width * self.channels
		y_test=pd.read_csv(self.testPath+filename)
		labels=y_test['Path'].values
		y_test=y_test['ClassId'].values
		y_test=np.array(y_test)

		d=[]
		for j in labels:
			i1=cv2.imread(self.testPath+j)
			i2=Image.fromarray(i1,'RGB')
			i3=i2.resize((self.height, self.width))
			d.append(np.array(i3))

		X_test=np.array(d)
		X_test = X_test.astype('float32')/255 
		print("Test :", X_test.shape, y_test.shape)
		return X_test, y_test


	def load_dataset(self, mode=None):
		data = []

		if self.dataset_name == 'MNIST':
			self.num_classes = 10
			self.channels = 1
			if mode == 'train':
				X_train, Y_train, X_valid, Y_valid, _, _, _ = self.load_mnist(self.validation_size)
				data = X_train, Y_train, X_valid, Y_valid
			else:
				_, _, _, _, X_test, y_test, _ = self.load_mnist(onehotencoder=False)
				data = X_test, y_test

		elif self.dataset_name == 'GTSRB':
			self.num_classes = 43
			self.channels = 3
			if mode == 'train':
				self.trainPath = 'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
				X_train, X_valid, Y_train, Y_valid = self.load_GTRSB_dataset(self.validation_size)
			else:
				self.testPath = 'data'+self.sep+'GTS_dataset'+self.sep+"kaggle"+self.sep
				X_test, y_test = self.load_GTRSB_csv("Test.csv")
				data = X_test, y_test

		elif self.dataset_name == 'cifar10':
			if mode == 'train':
				pass  
			else:
				pass

		else:
			print("Dataset not found!!")

		return data