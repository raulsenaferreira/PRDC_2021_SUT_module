import os
import keras
from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
import pandas as pd
from PIL import Image
import gzip
import cv2
import skimage.data
import skimage.transform
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class Dataset:
	"""docstring for Dataset"""
	def __init__(self, root_dir='data'):
		super(Dataset, self).__init__()
		self.dataset_name = ''
		self.modification = ''
		self.root_dir = root_dir
		self.width = 28
		self.height = 28
		self.channels = 0
		self.testPath = ''
		self.num_classes = 0
		self.trainPath = ''
		self.testPath = ''
		self.validation_size = None
		self.X = [[]]
		self.y = []

	
	def load_dataset(self, dataset_name, threat_name):

		train_images = os.path.join(self.root_dir, dataset_name, threat_name, 'train-images-npy.gz')
		train_labels = os.path.join(self.root_dir, dataset_name, threat_name, 'train-labels-npy.gz')
		
		test_images = os.path.join(self.root_dir, dataset_name, threat_name, 'test-images-npy.gz')
		test_labels = os.path.join(self.root_dir, dataset_name, threat_name, 'test-labels-npy.gz')

		f = gzip.GzipFile(train_images, "r")
		x_train = np.load(f)
		#x_train = np.frombuffer(x_train)#, dtype=i.dtype
		#x_train = np.fromfile(f)
		
		f = gzip.GzipFile(train_labels, "r")
		y_train = np.load(f)

		f = gzip.GzipFile(test_images, "r")
		x_test = np.load(f)

		f = gzip.GzipFile(test_labels, "r")
		y_test = np.load(f)
		
		#print("load_drift_mnist: ", x_train.shape)

		return (x_train, y_train), (x_test, y_test)