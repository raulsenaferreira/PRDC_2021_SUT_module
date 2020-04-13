from src.utils import util
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Model


def run(validation_size, batch_size, models_folder, epochs, model_name, sep, script_path):
	trainPath = str(script_path)+sep+'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
	#loading German traffic sign dataset
	X_train,X_valid,Y_train,Y_valid = util.load_GTRSB_dataset(trainPath, validation_size)

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

	return history