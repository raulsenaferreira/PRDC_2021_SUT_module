'''
print("processing abstract box\n Finding best K for clustering...")
#apply clustering
K = 1
kmeans = KMeans(n_clusters=K, random_state=0).fit(arrWeights)
inertia = kmeans.inertia_
threshold = 0.3 #reported threshold for the GTRSB dataset in the outside-of-box paper
while inertia > threshold:
	K+=1
	kmeans = KMeans(n_clusters=K, random_state=0).fit(arrWeights)
	inertia = kmeans.inertia_
	print("K and inertia:", K, inertia)

print("Clustering...")
clusters = kmeans.predict(arrWeights)
'''




'''
#testing
v0 = [-2, 1, 0, 3, 1]
v1 = [0, 3, 1, -2, 2]
v2 = [1, 0, -2, -3, 3]
v3 = [-1, -1, 2, 0, -1]
v4 = [3, -3, -1, 2, -2]
v5 = [-3, -2, -3, 3, 0]

data = np.asarray([v0, v1, v2, v3, v4, v5])
print("actv function", data)
#print("2D projection", data[:,[0,-1]])

K = 1
kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
inertia = kmeans.inertia_
threshold = 50 #reported threshold for the GTRSB dataset in the out-of-box paper
while inertia > threshold:
	K+=1
	kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
	inertia = kmeans.inertia_
	print("K and inertia:", K, inertia)

array_box = []
for i in range(data.shape[1]):
	min_i = np.amin(data[:,i])
	max_i = np.amax(data[:,i])
	array_box.append([min_i, max_i])
print("activ func bounds:", array_box)

clusters = kmeans.predict(data)
print("clusters activ func:", clusters)

#doing a projection by taking just the first and the last dimension of data
data = data[:,[0,-1]]

dataByCluster={}
for c, d in zip(clusters, data):
	#print(d)
	try:
		dataByCluster[c].append(d)
	except:
		dataByCluster.update({c:[d]})
#print(np.asarray(dataByCluster[0]))



array_box_by_cluster = {}

classe = 14
array_box_by_cluster.update({classe:[]})

for k, v in dataByCluster.items():
	arr_intermediate = []
	v = np.asarray(v)
	for i in range(v.shape[1]):
		min_i = np.amin(v[:,i])
		max_i = np.amax(v[:,i])
		arr_intermediate.append([min_i, max_i])
	array_box_by_cluster[classe].append(arr_intermediate)
	
	print("bounds of clustered ({}) activ func: {}".format(k, array_box_by_cluster[classe]))


test = np.asarray([[-3, -2, -3, 3, 0], [-9, -8, -7, 9, 5]])
#after clustering ...
test = test[:,[0,-1]]
for X in test:
	if not find_point(array_box_by_cluster[classe], X[0], X[1]):
		print("outside of the box!", X)

'''