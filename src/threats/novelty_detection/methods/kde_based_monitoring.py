import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import cv2 as cv
from skimage.color import rgb2gray
from src.Classes.dataset import Dataset


def crop_center(img,cropx,cropy):
	y,x = 28, 28
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)	
	return img[starty:starty+cropy,startx:startx+cropx]


def get_pdf_cuttoff(dataset_name, classes_to_monitor, cutoff='minimum', crop=True, grayscale=False):
	arr_pdf_cuttoff_by_class = []
	arr_PCA_by_class = []
	arr_kde_by_class = []

	dataset = Dataset(dataset_name)
	X, y, x_val, y_val = dataset.load_dataset(mode='train')
	y = np.argmax(y, axis=1) #if using training data
	y_val = np.argmax(y_val, axis=1) #if using training data

	X = np.vstack([X, x_val])
	y = np.hstack([y, y_val])

	for class_to_monitor in range(classes_to_monitor):
		X_croped = []
		indices = np.where(y == class_to_monitor)
		#data_reshaped = X[indices].flatten().reshape(X[indices].shape[0], -1)

		if crop:
			#crop all images in the center
			for img in X[indices]:
				imc = crop_center(img, 12, 12)
				if grayscale:
					X_croped.append(rgb2gray(imc))
				else:
					X_croped.append(imc)

			X_croped = np.asarray(X_croped)
			data_reshaped = X_croped.flatten().reshape(X_croped.shape[0], -1)

		# PCA fit for training
		pca = PCA(n_components=20, whiten=False)
		data_reshaped = pca.fit_transform(data_reshaped)

		# use grid search cross-validation to optimize the bandwidth
		params = {'bandwidth': np.logspace(-1, 1, 20)}
		grid = GridSearchCV(KernelDensity(), params)
		grid.fit(data_reshaped)
		
		#print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

		# use the best estimator to compute the kernel density estimate
		kde = grid.best_estimator_

		pdfs = np.exp(kde.score_samples(data_reshaped))
		#print("printing pdfs cutoff: min ({}) and max ({}) for the class {}".format(np.min(pdfs), np.max(pdfs), class_to_monitor))
		
		if cutoff == 'minimum':
			arr_pdf_cuttoff_by_class.append(np.min(pdfs))
		elif cutoff == 'maximum':
			arr_pdf_cuttoff_by_class.append(np.max(pdfs))

		arr_PCA_by_class.append(pca)
		arr_kde_by_class.append(kde)

	return arr_pdf_cuttoff_by_class, arr_kde_by_class, arr_PCA_by_class