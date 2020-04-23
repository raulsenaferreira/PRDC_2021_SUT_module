import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def make_abstraction(data, clusters, classe, dim_reduc_obj=None, dim_reduc_method='', monitors_folder=None):
	data = np.asarray(data)
	
	if dim_reduc_obj==None:
		#doing a projection by taking just the first and the last dimension of data
		data = data[:,[0,-1]]
	else:
		#using a dimensionality reduction function
		method = dim_reduc_obj.fit(data)
		print("Saving trained dim reduc method")
		pickle.dump(method, open(monitors_folder+dim_reduc_method+'_trained.p', "wb"))
		data = method.transform(data)

	print(data.shape)

	dataByCluster={}

	for c, d in zip(clusters, data):
		try:
			dataByCluster[c].append(d)
		except:
			dataByCluster.update({c:[d]})

	array_box_by_cluster = {}
	array_box_by_cluster.update({classe:[]})

	for k, v in dataByCluster.items():
		arr_intermediate = []
		v = np.asarray(v)

		for i in range(v.shape[1]):
			min_i = np.amin(v[:,i])
			max_i = np.amax(v[:,i])
			arr_intermediate.append([min_i, max_i])
		array_box_by_cluster[classe].append(arr_intermediate)
	return array_box_by_cluster


def find_point(boxes, intermediateValues, class_to_monitor, dim_reduc_obj=None):
	data = np.asarray(intermediateValues)
	#print(intermediateValues)
	
	if dim_reduc_obj!=None:
		#using a dimensionality reduction function
		data = dim_reduc_obj.transform(data.reshape(1, -1))[0]
		
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


def find_point_box_ensemble(arr_boxes, intermediateValues_all):
    result = False
    for i in range(len(intermediateValues_all)):
    	#print(i)
    	if i != 3: #CNN 3 with problem
	        data = np.asarray(intermediateValues_all[i])
	        boxes = arr_boxes[i]
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