import os
import numpy as np


def load_artifact(artifact_name, neptune_experiments):	
	result_arr = []
	tmp_path = os.path.join('results', 'temp')

	for experiment in neptune_experiments:
		experiment.download_artifact(artifact_name, tmp_path)
		file_path = os.path.join(tmp_path, artifact_name)
		arr = np.load(file_path)
		os.remove(file_path)

		result_arr.append(arr)

	return np.array(result_arr)


def save_results(experiment, arr_readouts, plot=False):
	print("saving experiments", experiment.name)
	filenames = config_ND.load_file_names()
	csvs_folder_path = os.path.join('src', 'tests', 'results', 'csv', experiment.sub_field, experiment.name)
	img_folder_path = os.path.join('src', 'tests', 'results', 'img', experiment.sub_field, experiment.name)

	metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(experiment.name, arr_readouts, img_folder_path+'all_images.pdf')