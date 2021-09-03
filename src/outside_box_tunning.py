import os
import numpy as np
from utils import util
from Classes.dataset import Dataset
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


'''
dataset_name = 'GTSRB' # best K = 17
num_classes = 43
from keras.models import load_model
backend = "keras"
'''

dataset_name = 'CIFAR-10' # best k = 5 to 10
num_classes = 10
from tensorflow.keras.models import load_model
backend = "tensorflow"

perc_of_data = 1

dataset_folder = os.path.join('C:', '\\Users', 'rsenaferre', 'Desktop', 'GITHUB', 'phd_data_generation', 'data', 'training_set')
path = os.path.join(dataset_folder, 'novelty_detection', dataset_name)
dataset = Dataset(dataset_name)
dataset.validation_size = 0.3
(x_train, y_train), (_, _) = dataset.load_dataset(path)
X, y = x_train[:int(len(x_train)*perc_of_data)], y_train[:int(len(y_train)*perc_of_data)]

#path to load the model
models_folder = os.path.join("src", "bin", "models", backend)
model_file = os.path.join(models_folder, 'leNet_{}.h5'.format(dataset_name))

model = load_model(model_file)

layer_index = -2 #last hidden layer
intermediateValues = []
label_intermediate_values = []

for img, lbl in zip(X, y):
	img = np.asarray([img])
	yPred = np.argmax(model.predict(img))

	if yPred == lbl:
		intermediateValues.append(util.get_activ_func(backend, model, img, layer_index)[0])
		#label_intermediate_values.append(lbl)

kmeans = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(kmeans, k=(2,num_classes), timings=False)
visualizer.fit(np.array(intermediateValues))	# Fit the data to the visualizer
visualizer.show()		# Finalize and render the figure