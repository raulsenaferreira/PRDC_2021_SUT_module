from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Model

class LeNet:
	"""docstring for LeNet"""
	def __init__(self, is_classification, num_classes):
		super(LeNet, self).__init__()
		self.is_classification = is_classification
		self.num_classes = num_classes
		
	def train(self, X_train, X_valid, Y_train, Y_valid, batch_size, epochs):
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
		model.add(Dense(self.num_classes, activation='softmax'))

		#Compilation of the model
		if self.is_classification:
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		else:
			#changing here for regression properties
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		#training
		history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))

		return model, history