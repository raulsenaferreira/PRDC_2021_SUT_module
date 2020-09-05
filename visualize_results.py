import os
import neptune
from src.Classes.readout import Readout
from src.utils import metrics
from src.utils import util
from src.Classes.dataset import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from keras.models import load_model
import seaborn as sns
from PIL import Image
from src.novelty_detection.methods import image_dist_matching as idm


sns.set()
sep = util.get_separator()

def visualize_experiments(id_experiments, names, title, classes_to_monitor):

	project = neptune.init('raulsenaferreira/PhD')
	experiments = project.get_experiments(id=id_experiments)

	arr_readouts = []
	img_folder_path = 'src'+sep+'tests'+sep+'results'+sep+'img'+sep 

	for experiment, name in zip(experiments, names):
		avg_cf = {}

		logs = experiment.get_logs()
		#print(logs['True Positive - Class 0'].y) 

		# storing results
		readout = Readout()
		readout.name = name
		
		readout.avg_acc = logs['Accuracy'].y
		readout.avg_time = logs['Process time'].y
		readout.avg_memory = logs['Memory'].y
		readout.avg_F1 = logs['F1'].y

		for class_to_monitor in range(classes_to_monitor):
			fp = 'False Positive - Class {}'.format(class_to_monitor)
			fn = 'False Negative - Class {}'.format(class_to_monitor)
			tp = 'True Positive - Class {}'.format(class_to_monitor)
			tn = 'True Negative - Class {}'.format(class_to_monitor)

			avg_cf.update({class_to_monitor: [int(float(logs[fp].y)), int(float(logs[fn].y)), int(float(logs[tp].y)), int(float(logs[tn].y))]})
		readout.avg_cf = avg_cf

		arr_readouts.append(readout)

	fig_name = img_folder_path+'all_methods_class_'+title+'.pdf'
	os.makedirs(img_folder_path, exist_ok=True)
	metrics.plot_pos_neg_rate_stacked_bars_total(title, arr_readouts, classes_to_monitor, fig_name)


def plot_images(data, labels, num_row, num_col):

	fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
	for i in range(num_row*num_col):
		try:
			ax = axes[i//num_col, i%num_col]
			ax.imshow(np.squeeze(data[i]), cmap='gray')
			ax.set_title('{}'.format(labels[i]))
			ax.set_axis_off()
		except Exception as e:
   			pass 	
		
	plt.tight_layout(pad=3.0)
	plt.show()


def plot_distribution(data):

	tsne = TSNE(n_components=2).fit_transform(data)

	# scale and move the coordinates so they fit [0; 1] range
	def scale_to_01_range(x):
		# compute the distribution range
		value_range = (np.max(x) - np.min(x))

		# move the distribution so that it starts from zero
		# by extracting the minimal value from all its values
		starts_from_zero = x - np.min(x)

		# make the distribution fit [0; 1] by dividing by its range
		return starts_from_zero / value_range

	# extract x and y coordinates representing the positions of the images on T-SNE plot
	tx = tsne[:, 0]
	ty = tsne[:, 1]

	tx = scale_to_01_range(tx)
	ty = scale_to_01_range(ty)

	# initialize a matplotlib plot
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# for every class, we'll add a scatter plot separately
	for label in colors_per_class:
		# find the samples of the current class in the data
		indices = [i for i, l in enumerate(labels) if l == label]

		# extract the coordinates of the points of this class only
		current_tx = np.take(tx, indices)
		current_ty = np.take(ty, indices)

		# convert the class color to matplotlib format
		color = np.array(colors_per_class[label], dtype=np.float) / 255

		# add a scatter plot with the corresponding color and label
		ax.scatter(current_tx, current_ty, c=color, label=label)

	# build a legend using the labels we set previously
	ax.legend(loc='best')

	# finally, show the plot
	plt.show()


def act_func(model, X):
	arrWeights = []

	for img in X:
		img = np.asarray([img])
		arrWeights.append(util.get_activ_func(model, img, layerIndex=-2)[0])

	return arrWeights


def visualize_distributions(dataset):
	dataset = Dataset(dataset_name)
	#X, y, _, _ = dataset.load_dataset(mode='train')
	#y = np.argmax(y, axis=1) #if using training data

	X, y = dataset.load_dataset(mode='test')
	num_row = 5
	num_col = 9

	#indices = np.unique(y, return_index=True)[1]
	indices = np.where(y < 20)
	#print(indices)
	'''
	#plot_images(X[indices], y[indices], num_row=num_row, num_col=num_col)


	dataset_name = 'BTSC'#'BTSC', GTSRB
	dataset = Dataset(dataset_name)
	X, y, _, _ = dataset.load_dataset(mode='train')
	y = np.argmax(y, axis=1) #if using training data

	num_row = 7
	num_col = 9

	indices = np.unique(y, return_index=True)[1]
	#print(indices)
	#plot_images(X[indices], y[indices], num_row=num_row, num_col=num_col)
	'''

	#path to load the model
	models_folder = "src"+sep+"bin"+sep+"models"+sep
	model_file = models_folder+'leNet_'+dataset_name+'.h5'

	# loading model
	model = load_model(model_file)
	components = 20
	weights = act_func(model, X[indices])
	pca_50 = PCA(n_components=components)
	pca_result_50 = pca_50.fit_transform(weights)
	print('Cumulative explained variation for {} principal components: {}'.format(components, np.sum(pca_50.explained_variance_ratio_)))

	#ploting distributions
	df_subset = {}

	'''
	#using PCA + TSNE...
	tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
	tsne_pca_results = tsne.fit_transform(pca_result_50)
	df_subset['one'] = tsne_pca_results[:,0]
	df_subset['two'] = tsne_pca_results[:,1]

	# or PCA + Isomap
	isomap = Isomap(n_components = 2)
	isomap_pca_results = isomap.fit_transform(pca_result_50)
	df_subset['one'] = isomap_pca_results[:,0]
	df_subset['two'] = isomap_pca_results[:,1]
	'''
	# or just Isomap
	isomap = Isomap(n_components = 2)
	isomap_results = isomap.fit_transform(weights)
	df_subset['one'] = isomap_results[:,0]
	df_subset['two'] = isomap_results[:,1]


	df_subset['y'] = y[indices]
	ax = sns.scatterplot(
		x="one", y="two",
		hue="y",
		palette=sns.color_palette("hls", 20),
		data=df_subset,
		legend="full"#, alpha=0.3,	ax=ax3
	)
	plt.show()



dataset_name = 'GTSRB'#'BTSC', GTSRB
classes_to_monitor = 43

# Get list of experiments
#experiments = project.get_experiments(id=['PHD-24', 'PHD-25', 'PHD-26'])
#names = ['oob', 'oob_isomap', 'oob_pca']

#experiments = project.get_experiments(id=['PHD-24', 'PHD-30', 'PHD-33'])
#names = ['oob_3_clusters', 'oob_1_cluster', 'oob_1_cluster V2']

#experiments = ['PHD-30', 'PHD-33', 'PHD-31', 'PHD-34', 'PHD-32', 'PHD-35', 'PHD-37', 'PHD-36']
#names = ['oob', 'oob V2', 'oob_isomap', 'oob_isomap V2', 'oob_pca', 'oob_pca V2', 'oob_pca_isomap', 'oob_pca_isomap V2']

#experiments = ['PHD-38', 'PHD-39', 'PHD-40', 'PHD-41', 'PHD-42', 'PHD-43'] #out of distribution experiments using 30% of BTSC dataset
#names = ['oob 1 cluster', 'oob 3 clusters', 'oob_isomap 1 cluster', 'oob_isomap 3 clusters', 'oob_pca 1 cluster', 'oob_pca 3 clusters']

#experiments = ['PHD-45', 'PHD-46', 'PHD-47', 'PHD-48', 'PHD-49', 'PHD-50', 'PHD-51', 'PHD-52'] #out of distribution experiments using 100% of BTSC dataset
#names = ['oob 1 cluster', 'oob 3 clusters', 'oob_isomap 1 cluster', 'oob_isomap 3 clusters', 'oob_pca 1 cluster', 'oob_pca 3 clusters', 'oob_pca_isomap 1 cluster', 'oob_pca_isomap 3 clusters']

experiments = ['PHD-45', 'PHD-54']
names = ['oob 1', 'oob 2']

total_instances = 19725
dataset = Dataset(dataset_name)
X, y = dataset.load_dataset(mode='test')

indices = np.where(y == 1)
image = np.asarray(X[indices][:50])
indices = np.where(y == 2)
reference = np.asarray(X[indices][:50])
#idm.plot_diff_images(image, reference)
print(image.shape, reference.shape)
idm.histograms(image, reference)
#sim = idm.compare_histograms(image, reference)
#visualize_experiments(experiments, names, 'ID=GTSRB; OOD=BTSC', classes_to_monitor)


'''
dataset = Dataset(dataset_name)
#X, y, _, _ = dataset.load_dataset(mode='train')
#y = np.argmax(y, axis=1) #if using training data

X, y = dataset.load_dataset(mode='test')
img = np.asarray([X[0]])

#path to load the model
models_folder = "src"+sep+"bin"+sep+"models"+sep
model_file = models_folder+'leNet_'+dataset_name+'.h5'

# loading model
model = load_model(model_file)

arrWeights = util.get_activ_func(model, img, layerIndex=0)[0]
print(np.shape(arrWeights))
plt.matshow(img[0, :, :, :])
plt.show()
plt.matshow(arrWeights[ :, :, np.shape()], cmap='viridis')
plt.show()
'''
'''
color = ('r','g','b')
for channel,col in enumerate(color):
    histr = cv2.calcHist([img],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale picture for class {}'.format(y[0]))
plt.show()

cv2.destroyAllWindows()
'''