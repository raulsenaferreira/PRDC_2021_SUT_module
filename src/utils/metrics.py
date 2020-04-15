from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from math import sqrt
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(y_actual, y_predicted):
    return round(accuracy_score(y_actual, y_predicted), 4)*100
    

def F1(arr_y_true, arr_y_predicted, average):
	arrF1 = []
	m = min(len(arr_y_true), len(arr_y_predicted))

	for y_true, y_pred in zip(arr_y_true, arr_y_predicted):
		y_true, y_pred = y_true[:m], y_pred[:m]
		arrF1.append(f1_score(y_true, y_pred, average=average))

	return arrF1


def stacked_bars(arrFP, arrFN, arrTP, arrTN):
	#N = 5
	#menMeans = (20, 35, 30, 35, 27)
	#womenMeans = (25, 32, 34, 20, 25)
	#menStd = (2, 3, 4, 1, 2)
	#womenStd = (3, 5, 2, 3, 3)
	#ind = np.arange(N)    # the x locations for the groups
	#width = 0.35       # the width of the bars: can also be len(x) sequence

	N = 

	p1 = plt.bar(ind, menMeans, width, yerr=menStd)
	p2 = plt.bar(ind, womenMeans, width,
	             bottom=menMeans, yerr=womenStd)

	plt.ylabel('Scores')
	plt.title('Scores by group and gender')
	plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
	plt.yticks(np.arange(0, 81, 10))
	plt.legend((p1[0], p2[0]), ('Men', 'Women'))

	plt.show()