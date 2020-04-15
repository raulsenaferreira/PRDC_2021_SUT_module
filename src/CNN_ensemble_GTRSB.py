import numpy as np
from src.utils import util
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Model


def run(validation_size, batch_size, models_folder, epochs, model_name_prefix, sep, script_path):
	arr_history = []
	trainPath = str(Path(script_path).parent)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
	#loading German traffic sign dataset
	X_train,X_valid,Y_train,Y_valid = util.load_GTRSB_dataset(trainPath, validation_size)
	array_function=[util.image_adjustment, util.histogram_equalization, util.adaptive_hist_eq, util.contrast_normalization]

	for i in range(len(array_function)+1):

		if i < len(array_function):
			X_train = np.array(list(map(array_function[i], X_train)))
			X_valid = np.array(list(map(array_function[i], X_valid)))

		#model building
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
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		#training
		arr_history.append(model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid)))
		#saving model
		model.save(models_folder+model_name_prefix+str(i)+'.h5')

	return arr_history