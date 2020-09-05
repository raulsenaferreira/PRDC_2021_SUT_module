import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from math import sqrt
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score
from src.utils import util


sep = util.get_separator()

def save_results(arr_readouts, csvs_folder_path, filenames, delimiter):
	[index_file, acc_file_name, cf_file_name, time_file_name, mem_file_name, f1_file_name] = filenames
	index_results = []
	results = [[], [], [], [], []]
	# CSVs
	for readout in arr_readouts:
		index_results.append(readout.name)
		results[0].append(readout.avg_acc) 
		results[1].append(readout.avg_time) 
		results[2].append(readout.avg_memory)
		results[3].append(readout.avg_F1)
		results[4].append(readout.avg_cf)

	os.makedirs(csvs_folder_path, exist_ok=True)
	np.savetxt(csvs_folder_path+index_file, [p for p in index_results], delimiter=delimiter, fmt='%s')	
	np.savetxt(csvs_folder_path+acc_file_name, [p for p in results[0]], delimiter=delimiter, fmt='%s')
	np.savetxt(csvs_folder_path+time_file_name, [p for p in results[1]], delimiter=delimiter, fmt='%s')
	np.savetxt(csvs_folder_path+mem_file_name, [p for p in results[2]], delimiter=delimiter, fmt='%s')
	np.savetxt(csvs_folder_path+f1_file_name, [p for p in results[3]], delimiter=delimiter, fmt='%s')
	np.savetxt(csvs_folder_path+cf_file_name, [p for p in results[4]], delimiter=delimiter, fmt='%s')


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


def plot_pos_neg_rate_stacked_bars_total(experiment_name, arr_readouts, num_classes, fig_path):
	figures = []
	x = []
	y_fp = [] 
	y_fn = [] 
	y_tp = [] 
	y_tn = []

	#COLOR = 'black'
	#mpl.rcParams['text.color'] = 'white'
	#mpl.rcParams['axes.labelcolor'] = 'black'
	#mpl.rcParams['xtick.color'] = 'black'
	#mpl.rcParams['ytick.color'] = 'black'
	mpl.rcParams['font.size'] = 12

	for readout in arr_readouts:
		fp, fn, tp, tn = 0, 0, 0, 0
		x.append(readout.name)

		for monitored_class in range(num_classes):
			fp += readout.avg_cf[monitored_class][0]
			fn += readout.avg_cf[monitored_class][1]
			tp += readout.avg_cf[monitored_class][2]
			tn += readout.avg_cf[monitored_class][3]

		plot_statistics(readout.name, tn, tp, fp, fn)

		y_fp.append(fp)
		y_fn.append(fn)
		y_tp.append(tp)
		y_tn.append(tn)

	xticks = [i for i in range(len(x))]
	
	fig = plt.figure()
	ax = fig.add_subplot()
	width = 0.3
	blue = [0, .4, .6]
	yellow = [1, 0.65, 0.25]
	red = [1, 0, 0]
	darkgrey = 'darkgrey'
	gray = 'gray'
	grey = 'grey'
	ax.bar(x, y_tp, color=darkgrey, edgecolor="white", width=width, label='True positive')
	sums = y_tp
	ax.bar(x, y_fn, bottom=sums, color=grey, edgecolor="white", hatch="x", width=width, label='False negative')
	sums =[_x + _y for _x, _y in zip(sums, y_fn)]
	ax.bar(x, y_fp, bottom=sums, color=gray, edgecolor='white', hatch=".", width=width, label='False positive')
	sums = [_x + _y for _x, _y in zip(sums, y_fp)]
	#ax.bar(x, y_tn, bottom=sums, color=[0, 0.2, 0.1], edgecolor='white', hatch="*", width=width, label='True negative')

	ax.set_xlabel("Methods")
	ax.set_ylabel("Instances")
	#ax.set_ylim([0, 100])
	ax.xaxis.set_ticks(xticks, x)
	ax.legend()
	#ax.annotate('{}'.format(height))

	for i in range(len(y_fp)):
		plt.annotate(str(y_tp[i]), xy=(width/2+i-0.2, y_tp[i]*0.2), va='bottom', ha='left')
		plt.annotate(str(y_fn[i]), xy=(width/2+i-0.2, (y_fn[i]+y_tp[i])-y_fn[i]*0.5), va='bottom', ha='left')
		plt.annotate(str(y_fp[i]), xy=(width/2+i-0.2, (y_fp[i]+y_fn[i]+y_tp[i])-y_fp[i]*0.5), va='bottom', ha='left')
		
	

	fig.suptitle(experiment_name)
	ax.figure.canvas.set_window_title(experiment_name)
	figures.append(fig)
	plt.show()

	multipage(fig_path, figures, dpi=250)


def multipage(filename, figs=None, dpi=200):
	pp = PdfPages(filename)
	if figs is None:
		figs = [plt.figure(n) for n in plt.get_fignums()]
	for fig in figs:
		fig.savefig(pp, format='pdf')
	pp.close()
	#usage
	#multipage('multipage_w_raster.pdf', [fig2, fig3], dpi=250)


def plot_act_func_boxes_animation(boxes, point):
	return True


def plot_statistics(title, tn, tp, fp, fn):
	print('Method:', title)
	print('fp', fp)
	print('fn', fn)
	print('tp', tp)
	print('tn', tn)

	total_instances = tn + tp + fp + fn
	#print("total instances = ", total_instances)

	tnr = tn/(tn+fp)
	tpr = tp/(tp+fn)
	#print("TNR @ TPR 95% =", )

	print("monitors accuracy =",(tn+tp)/(tn+tp+fp+fn))

	mcc = ((tp*tn) - (fp*fn)) / math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
	print("monitor MCC score = ", mcc)

	error = (fp+fn)/(tn+tp+fp+fn)
	print("monitors error =",error)

	confidence_score = 1.96 * math.sqrt((error * (1 - error)) / total_instances) # Wilson score with 95% confidence interval (1.96)
	print("confidence interval = [{}, {}]".format(error-confidence_score, error+confidence_score)) 

	precision = tp/(tp+fp)
	print("monitors precision =", precision)

	recall = tp/(tp+fn)
	print("monitors recall =", recall)

	F1 = 2 * (precision * recall) / (precision + recall)
	print("monitors F1 score =", F1)