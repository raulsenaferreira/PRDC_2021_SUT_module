from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from math import sqrt
import numpy as np
from sklearn.metrics import roc_auc_score


def save_results(results, file_name, sep):
	np.savetxt(file_name, [p for p in results], delimiter=sep, fmt='%s')


def evaluate(y_actual, y_predicted):
    return round(accuracy_score(y_actual, y_predicted), 4)*100
    

def F1(y_true, y_pred):
	return f1_score(y_true, y_pred, average=None)


def stacked_bars(dic_by_method):
	#N = 5
	#menMeans = (20, 35, 30, 35, 27)
	#womenMeans = (25, 32, 34, 20, 25)
	#menStd = (2, 3, 4, 1, 2)
	#womenStd = (3, 5, 2, 3, 3)
	ind = np.arange(len(dic_by_method)) # the x locations for the groups
	width = 0.35 # the width of the bars: can also be len(x) sequence

	for dic in dic_by_method:
		pass

	p1 = plt.bar(ind, menMeans, width)
	p2 = plt.bar(ind, womenMeans, width, bottom=menMeans)

	plt.ylabel('Scores')
	plt.title('Scores by group and gender')
	plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
	plt.yticks(np.arange(0, 81, 10))
	plt.legend((p1[0], p2[0]), ('Men', 'Women'))

	plt.show()