import os

'''
**OK** 1 = outside-of-box paper; 2 = outside-of-box using isomap instead of 2D projection;

**testing** 3 = outside-of-box with ensemble of DNN; 4 = same of 3 but using isomap strategy;

5 = same of 2 but using DBSCAN instead of KNN; 6 = same of 2 but clustering without dimension reduction;
7 = same of 5 but clustering without dimension reduction; 
8 = using the derivative of activation functions instead of raw values
'''


def glue_dataset_names(datasets, modifications):
	data = {}

	for d in datasets:
		data.update({d: []})

		for m in modifications:
			data[d].append('{}_{}'.format(m[0], m[1]))

	return data


def get_technique_params(technique):
	# Default
	PARAMS = {'use_alternative_monitor': False}# True = label -> act func; False = label -> act func if label == predicted
	PARAMS.update({'OOD_approach': 'equality'})
	PARAMS.update({'use_scaler': False})
	PARAMS.update({'grid_search': False})

	if 'sgd' == technique:
		PARAMS.update({'use_scaler': True})
		PARAMS.update({'grid_search': True})

	elif 'random_forest' ==  technique:
		PARAMS.update({'grid_search': True})

	elif 'ocsvm' == technique:
		PARAMS.update({'OOD_approach': 'outlier'})
	
	elif 'oob' in technique:
		PARAMS.update({'arr_n_clusters': [3]})
		PARAMS.update({'arr_n_components': [2]}) 
		PARAMS.update({'tau': [0.0001]}) 
		PARAMS.update({'OOD_approach': 'outside_of_box'})

	elif 'knn' == technique:
		PARAMS.update({'arr_n_clusters': [2, 3, 5, 10]})
		PARAMS.update({'use_scaler': True})
		
	elif 'hdbscan' == technique:
		PARAMS.update({'min_samples': [5, 10, 15]})  #min_samples 5, 10, 15
	
	return PARAMS


def get_experiment_params(setting_id):
	PARAMS = {}
	# directory of datasets
	#root_dir = os.path.join('D:','phd_data_generation','data', 'modified')
	root_dir = os.path.join('C:', '\\Users', 'rsenaferre', 'Desktop', 'GITHUB', 'phd_data_generation', 'data', 'modified')
	PARAMS.update({'root_dir': root_dir})

	if setting_id == 1:
		# begin data_params
		datasets = ['GTSRB'] # 'MNIST'
		PARAMS.update({'num_classes_to_monitor_ID': 43}) # 10

		#modifications = ['brightness_severity_1', 'brightness_severity_5']
		modifications = [('gtsrb', 'btsc')]

		PARAMS.update({'data': glue_dataset_names(datasets, modifications)})
		
		PARAMS.update({'ood_dataset_name': 'BTSC'})
		PARAMS.update({'num_classes_to_monitor_OOD': 62})
		# end data_params

		PARAMS.update({'model_names': ['leNet']}) # 'leNet', 'vgg16'

		PARAMS.update({'technique_names': ['oob']}) #'baseline', 'knn', 'ocsvm', 'random_forest', 'sgd', 'hdbscan', 'oob', 'oob_isomap', 'oob_pca', 'oob_pca_isomap'
	
	return PARAMS


# reading from the desc.txt in data folder
def get_data_params():
	pass