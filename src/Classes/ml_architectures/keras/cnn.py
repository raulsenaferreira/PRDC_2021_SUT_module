from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from src.Classes.classifier_impl import *

class DNN(Classifier_interface):
	"""docstring for CNN"""
	def __init__(self, num_classes):
		super(DNN, self).__init__()
		self.num_classes = num_classes	
	

	def train(self, x_train, y_train, x_valid, y_valid, batch_size, epochs, keras_pre_processing=False, train_iterator=None):
		# CNN
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
		                 activation='relu',
		                 input_shape=x_train.shape[1:]))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.num_classes, activation='softmax'))

		#training
		history = model.compile(loss=keras.losses.categorical_crossentropy,
		              optimizer=keras.optimizers.Adadelta(),
		              metrics=['accuracy'])

		if keras_pre_processing and train_iterator!=None:
			model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs)
		else:
			model.fit(x_train.reshape(len(x_train), 28, 28, 1), y_train, batch_size=batch_size, epochs=epochs, verbose=1,
				validation_data=(x_valid.reshape(len(x_valid), 28, 28, 1), y_valid))

		return model, history