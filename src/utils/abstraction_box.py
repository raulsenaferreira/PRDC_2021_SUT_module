import numpy as np


def make_abstraction(data, clusters, classe):
	data = np.asarray(data)
	#doing a projection by taking just the first and the last dimension of data
	data = data[:,[0,-1]]
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

def find_point(boxes, intermediateValues, class_to_monitor):
	data = np.asarray(intermediateValues)
	#print(intermediateValues)
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
			else : 
				result = False
	except:
		result = False
	return result


def find_point_box_ensemble(boxes, intermediateValues_all, class_to_monitor):
    result = False
    for i in range(len(intermediateValues_all)):
        data = np.asarray(intermediateValues_all[i])
        #print(intermediateValues_all)
        x = data[0]
        y = data[-1]
        #print("point:", x, y)
        
        try:
            for box in boxes:
                #B = box[0]
                #print(box)
                x1 = box[0][0]
                x2 = box[0][1]
                y1 = box[1][0]
                y2 = box[1][1]
                if x >= x1 and x <= x2 and y >= y1 and y <= y2: 
                    return True
                else : 
                    result = False
        except:
            result = False
    return result