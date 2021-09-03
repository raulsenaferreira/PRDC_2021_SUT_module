import unittest
import json
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely import geometry, affinity
from sklearn.cluster import KMeans
#from novelty_detection.utils import abstraction_box


class TestAbstractionBox(unittest.TestCase):
	
	def check_outside_of_box(self, boxes, intermediateValues, tau):
		is_outside_of_box = True
		data = np.asarray(intermediateValues)

		x = data[0]
		y = data[-1]

		point = Point(x, y)

		for polygon in boxes:
			box = list(zip(*polygon.exterior.coords.xy))
			print('first box',box)
			'''
			(x1, y1) = box[0]
			(x2, y1) = box[1]
			(x2, y2) = box[2]
			(x1, y2) = box[3]

			x1 = x1*tau-x1 if x1 <= 0 else x1-tau
			x2 = x2*tau+x2 if x2 > 0 else x2+tau
			y1 = y1*tau-y1 if y1 > 0 else y1-tau
			y2 = y2*tau+y2 if y2 > 0 else y2+tau

			rectangle = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
			print(rectangle)
			polygon = Polygon(rectangle)
			'''
			polygon = affinity.scale(polygon, xfact=tau, yfact=tau, origin='center')
			
			box = list(zip(*polygon.exterior.coords.xy))
			print('second box',box)

			if polygon.contains(point):
				print('point {} inside the box'.format(point))
				is_outside_of_box = False
		
		return is_outside_of_box


	def do_abstract_by_cluster(self,dataByCluster):
		arr_polygon = []

		for cluster, weights_neuron in dataByCluster.items():
			arr_boxes = []
			weights_neuron = np.asarray(weights_neuron)
	
			weights_neuron = weights_neuron[:,[0,-1]]
			
			x1 = np.amin(weights_neuron[:,0])
			x2 = np.amax(weights_neuron[:,0])
			y1 = np.amin(weights_neuron[:,1])
			y2 = np.amax(weights_neuron[:,1])

			# it works
			rectangle = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
			arr_polygon.append(Polygon(rectangle))


			#abstract_box = geometry.box(x1, y1, x2, y2)
			#arr_polygon.append(abstract_box)

			#print('abstract box area', abstract_box.area)
			#increased_abs_box = abstract_box.buffer(0.1)
			#print('abstract box increased area (10%)', increased_abs_box.area)
			#print('original', abstract_box.area)

		return arr_polygon
		

	def make_abstraction(self, data):
		dataByCluster={}

		clusters = KMeans(n_clusters=2).fit_predict(data)

		for c, d in zip(clusters, data):
			try:
				dataByCluster[c].append(d)
			except:
				dataByCluster.update({c:[d]})

		arr_polygon = self.do_abstract_by_cluster(dataByCluster)
		
		return arr_polygon


	def test_make_box(self):
		tau = 1.0001 # 1.1 for 10% of enlargement, 1.35 for 35% enlargement (range of values tested in the paper)
		#data_class0 = [[0.3, 0.45], [0.38, 0.51], [0.4, 0.48], [0.52, 0.48]] # from the paper
		#data_class1 = [[0.02, 0.33], [0.04, 0.3], [0, 0.27], [0, 0.3], [0, 0.39]] # from the paper
		v0 = [-2, 1, 0, 3, 1]
		v1 = [0, 3, 1, -2, 2]
		v2 = [1, 0, -2, -3, 3]
		v3 = [-1, -1, 2, 0, -1]
		v4 = [3, -3, -1, 2, -2]
		v5 = [-3, -2, -3, 3, 0]

		data = np.asarray([v0, v1, v2, v3, v4, v5])
		
		arr_polygon = self.make_abstraction(data)

		# intermediate values simul
		v99 = [-3, -2, -3, 3, 0]
		is_outside_of_box = self.check_outside_of_box(arr_polygon, v99, tau)
		self.assertEqual(is_outside_of_box, False, "Should be False")

		v99 = [-4, -3, -4, 4, 0]
		is_outside_of_box = self.check_outside_of_box(arr_polygon, v99, tau)
		self.assertEqual(is_outside_of_box, True, "Should be True")


if __name__ == '__main__':
	unittest.main()
