import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.cluster import KMeans
from src.utils import util
from matplotlib.path import Path



sep = util.get_separator()

def make_boxes_by_cluster(dataByCluster):
	array_box_by_cluster = []

	for k, v in dataByCluster.items():
		arr_boxes = []
		v = np.asarray(v)

		for i in range(v.shape[1]):
			min_i = np.amin(v[:,i])
			max_i = np.amax(v[:,i])

			arr_boxes.append([min_i, max_i])

		array_box_by_cluster.append(arr_boxes)

	return array_box_by_cluster


def make_boxes(data):
	arr_boxes = []

	for i in range(data.shape[1]):
		min_i = np.amin(data[:,i])
		max_i = np.amax(data[:,i])

		arr_boxes.append([min_i, max_i])

	return arr_boxes


def make_abstraction(data, monitor, save):
	data = np.asarray(data)
	#print('data.shape', data.shape)
	os.makedirs(monitor.monitors_folder, exist_ok=True)
	
	if monitor.dim_reduc_method==None:
		#doing a projection by taking just the first and the last dimension of data
		data = data[:,[0,-1]]
	else:
		#using a dimensionality reduction function
		if monitor.technique == 'oob_pca_isomap':
			pca_method = monitor.dim_reduc_method[0].fit(data)
			data1 = pca_method.transform(data)
			isomap_method = monitor.dim_reduc_method[1].fit(data1)
			data = isomap_method.transform(data1)

			if save:
				file_path = monitor.monitors_folder + monitor.dim_reduc_filename_prefix[0]
				print("Saving trained PCA in", file_path)
				pickle.dump(pca_method, open(file_path, "wb"))
				print("Saving trained Isomap in", file_path)
				file_path = monitor.monitors_folder + monitor.dim_reduc_filename_prefix[1]
				pickle.dump(isomap_method, open(file_path, "wb"))

		else:
			method = monitor.dim_reduc_method.fit(data)

			if save:
				print("Saving trained dim reduc method in", file_path)
				pickle.dump(method, open(file_path, "wb"))

			data = method.transform(data)
	
	print("making boxes...", data.shape)

	dataByCluster={}

	if monitor.n_clusters > 0:
		clusters = KMeans(n_clusters=monitor.n_clusters).fit_predict(data)

		for c, d in zip(clusters, data):
			try:
				dataByCluster[c].append(d)
			except:
				dataByCluster.update({c:[d]})

		return make_boxes_by_cluster(dataByCluster)

	else:
		return make_boxes(data)
	

def make_abstraction_without_dim_reduc(data, monitor, save):
	data = np.asarray(data)

	print(data.shape)

	dataByCluster={}
	clusters = KMeans(n_clusters=monitor.n_clusters).fit_predict(data)
	
	print("making boxes without reducing dimension...")

	for c, d in zip(clusters, data):
		try:
			dataByCluster[c].append(d)
		except:
			dataByCluster.update({c:[d]})

	array_box_by_cluster = {}
	array_box_by_cluster.update({monitor.class_to_monitor:[]})

	for k, v in dataByCluster.items():
		arr_boxes = []
		v = np.asarray(v)

		for i in range(v.shape[1]):
			min_i = np.amin(v[:,i])
			max_i = np.amax(v[:,i])
			arr_boxes.append([min_i, max_i])
		array_box_by_cluster[monitor.class_to_monitor].append(arr_boxes)

	return array_box_by_cluster


def find_point(boxes, intermediateValues, class_to_monitor, monitor_folder, dim_reduc_obj):
	ok = 0
	result = False
	data = np.asarray(intermediateValues)
	#print(np.shape(data))
	x,y = None, None
	
	if dim_reduc_obj!=None:
		if type(dim_reduc_obj) == type([]):
			dim_reduc_obj_1 = pickle.load(open(monitor_folder+str(class_to_monitor) +sep+'trained_'+dim_reduc_obj[0], "rb"))
			intermediate_data = dim_reduc_obj_1.transform(data.reshape(1, -1))[0]
			dim_reduc_obj_2 = pickle.load(open(monitor_folder+str(class_to_monitor) +sep+'trained_'+dim_reduc_obj[1], "rb"))
			data = dim_reduc_obj_2.transform(intermediate_data.reshape(1, -1))[0]
		else:
			dim_reduc_obj = pickle.load(open(monitor_folder+str(class_to_monitor) +sep+'trained_'+dim_reduc_obj+'.p', "rb"))
			#data = dim_reduc_obj[class_to_monitor].transform(data.reshape(1, -1))[0] #old version
			data = dim_reduc_obj.transform(data.reshape(1, -1))[0] #last version
			#data = dim_reduc_obj.transform(data)
			#print(np.shape(data))
		x = data[0]
		y = data[1]
	else:
		x = data[0]
		y = data[-1]
		#print("find_point:", x,y)
	#print(np.shape(boxes))

	try:
		for box in boxes:
			#B = box[0]
			#print(class_to_monitor, box)
			x1 = box[0][0]
			x2 = box[0][1]
			y1 = box[1][0]
			y2 = box[1][1]
			
			if x >= x1 and x <= x2 and y >= y1 and y <= y2: 
				if x==0 and y==0:
					ok = 1
				return True, ok
	except:
		pass
		#print("error @ find_point function")
	#print("point:", x, y)

	return result, ok


def find_point_2(boxes, intermediateValues, class_to_monitor, monitor_folder, dim_reduc_obj):
	
	#result = False
	data = np.asarray(intermediateValues)

	if dim_reduc_obj!=None:
		dim_reduc_obj = pickle.load(open(monitor_folder+str(class_to_monitor) +sep+'trained_'+dim_reduc_obj+'.p', "rb"))
		#data = dim_reduc_obj[class_to_monitor].transform(data.reshape(1, -1))[0] #old version
		data = dim_reduc_obj.transform(data.reshape(1, -1))[0] #last version
		#data = dim_reduc_obj.transform(data)
		#print(np.shape(data))

		#x = data[0]
		#y = data[1]
	else:
		x = data[0]
		y = data[-1]
		data = [x, y]
		data = np.reshape(data, (-1, 2))
		#print(np.shape(data))
		print("find_point_2:", data)

	path1 = Path(boxes[0])
	index = path1.contains_points(data)
	print("return:", index)

	#return result


def find_point_box_ensemble(arr_boxes, intermediateValues_all, dim_reduc_obj):
	result = False
	for i in range(len(intermediateValues_all)):
		#print(i)
		if i != 3: #CNN 3 with problem
			data = np.asarray(intermediateValues_all[i])
			boxes = arr_boxes[i]

		if dim_reduc_obj != None:
			data = dim_reduc_obj.transform(data.reshape(1, -1))[0]

			x = data[0]
			y = data[-1]
			#print("point:", x, y)
			try:
				for box in boxes:
					x1 = box[0][0]
					x2 = box[0][1]
					y1 = box[1][0]
					y2 = box[1][1]

					if x >= x1 and x <= x2 and y >= y1 and y <= y2:
						#print("similarity between 0 and 1", cosine_similarity(intermediateValues_all[0].reshape(1, -1), intermediateValues_all[1].reshape(1, -1)))
						#print("similarity between 0 and 2", cosine_similarity(intermediateValues_all[0].reshape(1, -1), intermediateValues_all[2].reshape(1, -1)))
						#print("similarity between 0 and 4", cosine_similarity(intermediateValues_all[0].reshape(1, -1), intermediateValues_all[4].reshape(1, -1)))
						#print("similarity between 1 and 2", cosine_similarity(intermediateValues_all[1].reshape(1, -1), intermediateValues_all[2].reshape(1, -1)))
						#print("similarity between 1 and 4", cosine_similarity(intermediateValues_all[1].reshape(1, -1), intermediateValues_all[4].reshape(1, -1)))
						#print("similarity between 2 and 4", cosine_similarity(intermediateValues_all[2].reshape(1, -1), intermediateValues_all[4].reshape(1, -1)))

						return True
			except:
				pass
					#print("error @ find_point_box_ensemble function")
	return result