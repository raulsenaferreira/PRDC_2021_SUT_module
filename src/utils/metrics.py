from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from math import sqrt
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score


def save_results(results, file_name, sep):
	np.savetxt(file_name, [p for p in results], delimiter=sep, fmt='%s')


def evaluate(y_actual, y_predicted):
	return round(accuracy_score(y_actual, y_predicted), 4)*100
	

def F1(y_true, y_pred):
	return f1_score(y_true, y_pred, average=None)


def plot_false_decisions_legend():
	fig = plt.figure()
	ax = fig.add_subplot()
	title = "Legend"
	ax.figure.suptitle(title)
	ax.figure.canvas.set_window_title(title)
	labels = 'false positives', 'false negatives', 'true positives'
	width = 0.5

	blue = [0, .4, .6]
	yellow = [1, 0.65, 0.25]
	red = [1, 0, 0]
	res1 = ax.bar([1], [1], color=red, edgecolor='white', hatch=".", width=width)
	res2 = ax.bar([1], [2], bottom=[1], color=yellow, edgecolor="white", hatch="x", width=width)
	res3 = ax.bar([1], [3], bottom=[2], color=blue, edgecolor="white", width=width)
	ax.cla()
	plt.axis('off')
	ax.legend((res1[0], res2[0], res3[0]), labels, loc="center", handleheight=3)
	plt.show()


def plot_pos_neg_rate_stacked_bars(confusion_matrices, datasets, fig_path):
	figures = []
	for i in range(len(datasets)):
		x = []
		y_fp = [] 
		y_fn = [] 
		y_tp = [] 
		y_tn = []

		for cf in confusion_matrices:
			x.append(cf[0])
			y_fp.append(cf[i+1][0])
			y_fn.append(cf[i+1][1])
			y_tp.append(cf[i+1][2])
			y_tn.append(cf[i+1][3])

		xticks = [i for i in range(len(x))]
		
		fig = plt.figure()
		ax = fig.add_subplot()
		width = 0.2
		blue = [0, .4, .6]
		yellow = [1, 0.65, 0.25]
		red = [1, 0, 0]
		ax.bar(x, y_tp, color=blue, edgecolor="white", width=width)
		sums = y_tp
		ax.bar(x, y_fn, bottom=sums, color='yellow', edgecolor="white", hatch="x", width=width)
		sums =[_x + _y for _x, _y in zip(sums, y_fn)]
		ax.bar(x, y_fp, bottom=sums, color=red, edgecolor='white', hatch=".", width=width)
		sums = [_x + _y for _x, _y in zip(sums, y_fp)]
		ax.bar(x, y_tn, bottom=sums, color=[0, 0.2, 0.1], edgecolor='white', hatch="*", width=width)

		ax.set_xlabel("Methods")
		ax.set_ylabel("Percentage")
		ax.set_ylim([0, 100])
		ax.xaxis.set_ticks(xticks, x)
		
		fig.suptitle(datasets[i])
		ax.figure.canvas.set_window_title(datasets[i])
		figures.append(fig)
		plt.show()

	multipage(fig_path, figures, dpi=250)

'''
def plot_pos_neg_rate_stacked_bars(scores):
	#N = 5
	#menMeans = (20, 35, 30, 35, 27)
	#womenMeans = (25, 32, 34, 20, 25)
	#menStd = (2, 3, 4, 1, 2)
	#womenStd = (3, 5, 2, 3, 3)
	ind = np.arange(len(dic_by_method)) # the x locations for the groups
	width = 0.35 # the width of the bars: can also be len(x) sequence

	p1 = plt.bar(ind, menMeans, width)
	p2 = plt.bar(ind, womenMeans, width)

	plt.ylabel('Scores')
	plt.title('Scores by group and gender')
	plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
	plt.yticks(np.arange(0, 81, 10))
	plt.legend((p1[0], p2[0]), ('Men', 'Women'))

	plt.show()
'''

def multipage(filename, figs=None, dpi=200):
	pp = PdfPages(filename)
	if figs is None:
		figs = [plt.figure(n) for n in plt.get_fignums()]
	for fig in figs:
		fig.savefig(pp, format='pdf')
	pp.close()
	#usage
	#multipage('multipage_w_raster.pdf', [fig2, fig3], dpi=250)


#confusion_matrices = [['CNN_OOB', [0.0, 11.0, 0.0, 1021.0], [14.0, 14.0, 1.0, 417.0]], ['CNN_OOB_isomap', [5.0, 11.0, 0.0, 1016.0], [7.0, 13.0, 2.0, 424.0]]]
#datasets = ['MNIST', 'GTSRB']
#img_name = 'all_images.pdf'
#img_folder_path = 'results'+sep+'img'+sep
#plot_pos_neg_rate_stacked_bars(confusion_matrices, datasets, img_folder_path+img_name)