import os
import logging
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from keras.models import Model
from keras.models import load_model
from IPython.display import Image, display
from src.Classes.model_builder import ModelBuilder
from src.Classes.dataset import Dataset
from src.utils import util
import pickle



def set_tf_loglevel(level):
	if level >= logging.FATAL:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	if level >= logging.ERROR:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	if level >= logging.WARNING:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
	else:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
	logging.getLogger('tensorflow').setLevel(level)


def compute_loss(model, input_image, filter_index):
	layer = model.get_layer(index=-2)
	feature_extractor = Model(inputs=model.inputs, outputs=layer.output)

	print("np.shape(input_image)", np.shape(input_image))

	activation = feature_extractor(input_image)
	#print(np.shape(activation))

	# We avoid border artifacts by only involving non-border pixels in the loss.
	filter_activation = activation[:, 2:-2, 2:-2, filter_index]
	return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(model, img, filter_index, learning_rate):
	with tf.GradientTape() as tape:
		tape.watch(img)
		loss = compute_loss(model, img, filter_index)
	# Compute gradients.
	grads = tape.gradient(loss, img)
	# Normalize gradients.
	grads = tf.math.l2_normalize(grads)
	img += learning_rate * grads
	return loss, img

def initialize_image(img_width, img_height):
	# We start from a gray image with some random noise
	img = tf.random.uniform((1, img_width, img_height, 3))
	# ResNet50V2 expects inputs in the range [-1, +1].
	# Here we scale our random inputs to [-0.125, +0.125]
	return (img - 0.5) * 0.25


def visualize_filter(model, filter_index, img):
	# We run gradient ascent for 20 steps
	iterations = 30
	learning_rate = 10.0
	#img = initialize_image(img_width, img_height)
	for iteration in range(iterations):
		loss, img = gradient_ascent_step(model, img, filter_index, learning_rate)

	# Decode the resulting input image
	img = deprocess_image(img[0].numpy())
	return loss, img


def deprocess_image(img):
	# Normalize array: center on 0., ensure variance is 0.15
	img -= img.mean()
	img /= img.std() + 1e-5
	img *= 0.15

	# Center crop
	img = img[25:-25, 25:-25, :]

	# Clip to [0, 1]
	img += 0.5
	img = np.clip(img, 0, 1)

	# Convert to RGB array
	img *= 255
	img = np.clip(img, 0, 255).astype("uint8")
	return img


if __name__ == "__main__":
	# disabling tensorflow logs
	set_tf_loglevel(logging.FATAL)
	# re-enabling tensorflow logs
	#set_tf_loglevel(logging.INFO)

	filter_index = 0
	img_width, img_height = 128, 128

	model_name = 'leNet'
	dataset_name = 'GTSRB'

	# loading model
	model = ModelBuilder()
	model = load_model(model.models_folder+model_name+'_'+dataset_name+'.h5')

	dataset = Dataset(dataset_name)
	X, y = dataset.load_dataset(mode='test')

	loss, img = visualize_filter(model, filter_index, X[0])

	keras.preprocessing.image.save_img("0.png", X[0])
	display(Image("0.png"))