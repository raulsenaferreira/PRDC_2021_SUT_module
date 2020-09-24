import os
from math import sqrt
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from src.utils import util
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report


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


def evaluate(y_true, y_pred, metric='accuracy'):
	if metric=='accuracy':
		return round(accuracy_score(y_true, y_pred), 4)*100
	elif metric=='F1':
		return f1_score(y_true, y_pred, average=None)
	elif metric=='MCC':
		return matthews_corrcoef(y_true, y_pred)
	elif metric=='precision':
		return precision_score(y_true, y_pred, average=None)
	elif metric=='recall':
		return recall_score(y_true, y_pred, average=None)
	else:
		print("Metric not found!")
	

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

	#print("TNR @ TPR 95% =", )

	print("monitors accuracy =",(tn+tp)/(tn+tp+fp+fn))

	mcc = ((tp*tn) - (fp*fn)) / math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
	print("monitor MCC score = ", mcc)

	error = (fp+fn)/(tn+tp+fp+fn)
	print("monitors error =",error)

	confidence_score = 1.96 * math.sqrt((error * (1 - error)) / total_instances) # Wilson score with 95% confidence interval (1.96)
	print("confidence interval = [{}, {}]".format(error-confidence_score, error+confidence_score)) 

	tnr = tn/(tn+fp) # specificity
	print("monitors specificity =", tnr)

	precision = tp/(tp+fp)
	print("monitors precision =", precision)

	recall = tp/(tp+fn) # sensitivity or TPR
	print("monitors recall =", recall)

	F1 = 2 * (precision * recall) / (precision + recall)
	print("monitors F1 score =", F1)
	

def ROC_ID_OOD(y_test_ID, y_score_ID, y_test_OOD, y_score_OOD):
	fpr_id, tpr_id, thresholds_id = roc_curve(y_test_ID, y_score_ID, drop_intermediate=False)
	fpr_ood, tpr_ood, thresholds_ood = roc_curve(y_test_OOD, y_score_OOD, drop_intermediate=False)

	return fpr_id, tpr_id, thresholds_id, fpr_ood, tpr_ood, thresholds_ood


def plot_ROC_curve_ID_OOD(list_of_readouts, mode):
	#(y_test_ID, y_score_ID, y_test_OOD, y_score_OOD)
	#y_test = np.hstack([y_test_ID, y_test_OOD])
	#y_score = np.hstack([y_score_ID, y_score_OOD])
	#fpr, tpr, _ = roc_curve(y_test, y_score)
	plt.figure()
	lw = 2
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.title('ROC Curves')
	

	for readout in list_of_readouts:
		
		y_test_ID = readout.arr_pos_neg_ID_true
		y_score_ID = readout.arr_pos_neg_ID_pred
		y_test_OOD = readout.arr_pos_neg_OOD_true
		y_score_OOD = readout.arr_pos_neg_OOD_pred
		title = readout.title
		target_names = ['Negative', 'Positive']
		print(metrics.classification_report(y_test_ID, y_score_ID, target_names=target_names))
		print(metrics.classification_report(y_test_OOD, y_score_OOD, target_names=target_names))

		disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
		disp.figure_.suptitle("Confusion Matrix")
		print("Confusion matrix:\n%s" % disp.confusion_matrix)

		plt.show()
	'''
		fpr_id, tpr_id, thresholds_id, fpr_ood, tpr_ood, thresholds_ood = ROC_ID_OOD(y_test_ID, y_score_ID, y_test_OOD, y_score_OOD)
	
		if mode == 'fp_tp':
			# fpr ood x tpr id 
			# True positive rate in ID over false positive rate in OOD = verifies how much the monitor
			# helps the classifier in avoiding missclassification on known data while avoiding raising false alarms on OOD
			roc_auc = auc(fpr_ood, tpr_id)
			label = '{} (area = {})'.format(title, round(roc_auc, 2))
			
			plt.plot(fpr_ood, tpr_id, #color='darkorange',
			         lw=lw, label=label)
			plt.legend(loc="lower right")

			plt.xlabel('FPR on {}'.format(readout.ood_dataset))
			plt.ylabel('TPR on {}'.format(readout.id_dataset))
			
		elif mode == 'tp_fp':
			# tpr ood x fpr id
			# False positive rate in ID over true positive rate in OOD = verifies how much the monitor
			# raises false alarms on known data while correctly identifying OOD data
			roc_auc = auc(tpr_ood, fpr_id)
			label = '{} (area = {})'.format(title, round(roc_auc, 2))
			
			plt.plot(tpr_ood, fpr_id, lw=lw, label=label)
			plt.legend(loc="lower right")

			plt.xlabel('TPR on {}'.format(readout.ood_dataset))
			plt.ylabel('FPR on {}'.format(readout.id_dataset))

		area = get_AUPR(y_test_ID, y_score_ID)
		print ("Area Under PR Curve(AP) ID: %0.2f" % area)
		print('AUROC ID', roc_auc_score(y_test_ID, y_score_ID))

		area = get_AUPR(y_test_OOD, y_score_OOD)
		print ("Area Under PR Curve(AP) OOD: %0.2f" % area)
		print('AUROC OOD', roc_auc_score(y_test_OOD, y_score_OOD))

		print('thresholds_ood', thresholds_ood)
		print('thresholds_id', thresholds_id)
		ind = np.where(tpr_ood==0.95)

		print('FPR at 95% TPR:', fpr_id[ind])

	plt.show()
	'''


def plot_FPR_at_TPR(fpr, tpr, tpr_rate=0.95):
	pass


def get_AUPR(labels, predicted):
	precision, recall, thresholds = precision_recall_curve(labels, predicted)
	area = auc(recall, precision)
	return area


def plot_box_analysis(boxes_monitor, point):
	pass


def plot_monitored_instances():
	pass


def confusion_matrix():
	import seaborn as sn
	import pandas as pd
	import matplotlib.pyplot as plt
	

	array = [[13,1,1,0,2,0],
	         [3,9,6,0,1,0],
	         [0,0,16,2,0,0],
	         [0,0,0,13,0,0],
	         [0,0,0,0,15,0],
	         [0,0,1,0,0,15]]

	df_cm = pd.DataFrame(array, range(6), range(6))
	plt.figure(figsize=(15,10))
	sn.set(font_scale=1) # for label size
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size

	plt.show()


if __name__ == '__main__':
	confusion_matrix()