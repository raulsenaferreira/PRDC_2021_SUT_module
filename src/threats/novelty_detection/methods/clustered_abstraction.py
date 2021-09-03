import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.cluster import KMeans
from src.utils import util
from matplotlib.path import Path



sep = util.get_separator()

def make_abstraction_without_dim_reduc(data, monitor, save):
	data = np.asarray(data)

	print(data.shape)

	clusters = KMeans(n_clusters=monitor.n_clusters).fit(data)
	dataByCluster={monitor.class_to_monitor:clusters}

	print("saving clustered activation function values...")

	return dataByCluster


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

	return result, ok