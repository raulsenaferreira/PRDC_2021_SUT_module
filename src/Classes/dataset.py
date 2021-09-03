import os
import numpy as np
from PIL import Image
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


class Dataset:
	"""docstring for Dataset"""
	def __init__(self, root_dir='data', transform=None):
		super(Dataset, self).__init__()
		self.dataset_name = ''
		self.modification = ''
		self.dataset_ID_name = ''
		self.dataset_OOD_name = ''
		self.root_dir = root_dir
		self.width = 0
		self.height = 0
		self.channels = 0
		self.testPath = ''
		self.num_classes = 0
		self.trainPath = ''
		self.testPath = ''
		self.validation_size = None
		self.X = [[]]
		self.y = []
		self.transform = transform


	def __getitem__(self, index):
		image = self.X[index]
		
		# Just apply your transformations here
		if self.transform is not None:
			image = self.transform(image)
		x = TF.to_tensor(image)
		
		return x, self.y[index]


	def __len__(self):
		return len(self.y)

	
	def load_dataset(self, dataset_path, mode='train'):
		x_train, y_train, x_test, y_test = None, None, None, None
		
		train_images = os.path.join(dataset_path, 'train-images-npy.gz')
		train_labels = os.path.join(dataset_path, 'train-labels-npy.gz')
	
		test_images = os.path.join(dataset_path, 'test-images-npy.gz')
		test_labels = os.path.join(dataset_path, 'test-labels-npy.gz')

		if mode == 'train' or mode == 'all':
			f = gzip.GzipFile(train_images, "r")
			x_train = np.load(f)
		
			f = gzip.GzipFile(train_labels, "r")
			y_train = np.load(f)

		elif mode == 'test' or mode == 'all':
			f = gzip.GzipFile(test_images, "r")
			x_test = np.load(f)

			f = gzip.GzipFile(test_labels, "r")
			y_test = np.load(f)

		# all images are already normalized (values divided by 255.)
		return (x_train, y_train), (x_test, y_test)