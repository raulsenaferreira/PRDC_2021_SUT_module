import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.cluster import KMeans


def make_boxes(class_to_monitor, dataByCluster):
	array_box_by_cluster = {}
	array_box_by_cluster.update({class_to_monitor:[]})

	for k, v in dataByCluster.items():
		arr_intermediate = []
		v = np.asarray(v)

		for i in range(v.shape[1]):
			min_i = np.amin(v[:,i])
			max_i = np.amax(v[:,i])
			arr_intermediate.append([min_i, max_i])
		array_box_by_cluster[class_to_monitor].append(arr_intermediate)

	return array_box_by_cluster


def make_abstraction(data, monitor):
	data = np.asarray(data)
	print('data.shape', data.shape)
	
	if monitor.dim_reduc_method==None:
		#doing a projection by taking just the first and the last dimension of data
		data = data[:,[0,-1]]
	else:
		#using a dimensionality reduction function
		method = monitor.dim_reduc_method.fit(data)
		file_path = monitor.monitors_folder + monitor.dim_reduc_filename_prefix

		os.makedirs(monitor.monitors_folder, exist_ok=True)
		
		print("Saving trained dim reduc method in", file_path)
		pickle.dump(method, open(file_path, "wb"))

		data = method.transform(data)

	dataByCluster={}
	clusters = KMeans(n_clusters=monitor.n_clusters).fit_predict(data)
	
	print("making boxes...", data.shape)

	for c, d in zip(clusters, data):
		try:
			dataByCluster[c].append(d)
		except:
			dataByCluster.update({c:[d]})

	return make_boxes(monitor.class_to_monitor, dataByCluster)


def make_abstraction_without_dim_reduc(data, monitor):
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
		arr_intermediate = []
		v = np.asarray(v)

		for i in range(v.shape[1]):
			min_i = np.amin(v[:,i])
			max_i = np.amax(v[:,i])
			arr_intermediate.append([min_i, max_i])
		array_box_by_cluster[monitor.class_to_monitor].append(arr_intermediate)

	return array_box_by_cluster


def find_point(boxes, intermediateValues, class_to_monitor, dim_reduc_obj):
	data = np.asarray(intermediateValues)
	#print(intermediateValues)
	
	if dim_reduc_obj!=None:
		data = dim_reduc_obj[class_to_monitor].transform(data.reshape(1, -1))[0]
		
	#print(data)
	x = data[0]
	y = data[-1]
	#print("point:", x, y)
	result = False
	try:
		for box in boxes[class_to_monitor]:
			#B = box[0]
			#print(box)
			x1 = box[0][0]
			x2 = box[0][1]
			y1 = box[1][0]
			y2 = box[1][1]
			if x >= x1 and x <= x2 and y >= y1 and y <= y2: 
				return True
	except:
		pass
		#print("error @ find_point function")
	return result


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