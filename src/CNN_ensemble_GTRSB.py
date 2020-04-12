import sys
import os
import numpy as np
from pathlib import Path
from utils import util
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Model


is_windows = sys.platform.startswith('win')
sep = '\\'

if is_windows == False:
    sep = '/'

script_path = os.getcwd()
trainPath = str(Path(script_path).parent)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
epochs = 10
batch_size = 32
validation_size = 0.3
models_folder = "bin"+sep+"models"+sep

#loading German traffic sign dataset
X_train,X_valid,Y_train,Y_valid = util.load_GTRSB_dataset(trainPath, validation_size)

array_function=[util.image_adjustment, util.histogram_equalization, util.adaptive_hist_eq, util.contrast_normalization]

for i in range(len(array_function)):
	model_name = 'CNN_ensemble_GTRSB_'+str(i)+'.h5'
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
	history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid))
	#saving model
	model.save(models_folder+model_name)