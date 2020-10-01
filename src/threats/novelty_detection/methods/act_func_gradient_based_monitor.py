import os
import numpy as np
import pickle
from src.utils import util
import tensorflow as tf
from tensorflow import keras



def compute_loss(input_image, filter_index):
	feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
	
	activation = feature_extractor(input_image)
	# We avoid border artifacts by only involving non-border pixels in the loss.
	filter_activation = activation[:, 2:-2, 2:-2, filter_index]
	return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
	with tf.GradientTape() as tape:
		tape.watch(img)
		loss = compute_loss(img, filter_index)
	# Compute gradients.
	grads = tape.gradient(loss, img)
	# Normalize gradients.
	grads = tf.math.l2_normalize(grads)
	img += learning_rate * grads
	return loss, img


def build_monitor(model, X, y, class_to_monitor, layer_index):
	arrWeights = []

	#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
	counter = 0
	loading_percentage = 0.1
	loaded = int(loading_percentage*len(y))

	for img, lab in zip(X, y):
		lab = np.where(lab)[0]
		counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
		img = np.asarray([img])
		yPred = np.argmax(model.predict(img))
		
		if util.is_monitored_prediction(yPred, lab, class_to_monitor):
			arrWeights.append(util.get_activ_func(model, img, layerIndex=layer_index)[0])

	return arrWeights


def build_monitor_without_classify(model, X, y, class_to_monitor, layer_index):
	arrWeights = []

	#comment these 3 lines and the line with "log" if you want turn off notification about loaded data 
	counter = 0
	loading_percentage = 0.1
	loaded = int(loading_percentage*len(y))

	for img, lab in zip(X, y):
		lab = np.where(lab)[0]
		counter, loading_percentage = util.loading_info(counter, loaded, loading_percentage) #log
		img = np.asarray([img])
		
		if lab == class_to_monitor:
			arrWeights.append(util.get_activ_func(model, img, layerIndex=layer_index)[0])

	return arrWeights


def run(monitor, model, X, y, save):
	class_to_monitor = monitor.class_to_monitor
	layer_index = monitor.layer_index

	#building monitor with training set
	arrWeights = build_monitor(model, X, y, class_to_monitor, layer_index)
	trained_monitor = monitor.method(arrWeights, monitor, save)

	file_path = monitor.monitors_folder+monitor.filename
	if save:
		print("Saving monitor in", file_path)
		os.makedirs(monitor.monitors_folder, exist_ok=True)
		pickle.dump(trained_monitor, open( file_path, "wb" ))


	#testing new way to build monitors
	arrWeights_2 = build_monitor_without_classify(model, X, y, class_to_monitor, layer_index)
	trained_monitor_2 = monitor.method(arrWeights_2, monitor, save)
	
	file_path = monitor.monitors_folder+monitor.filename+'_2'
	if save:
		print("Saving monitor 2 in", file_path)
		os.makedirs(monitor.monitors_folder, exist_ok=True)
		pickle.dump(trained_monitor_2, open( file_path, "wb" ))

	return trained_monitor, trained_monitor_2