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
	PARAMS.update({'use_scaler': False})
	PARAMS.update({'grid_search': False})

	if 'sgd' == technique:
		PARAMS.update({'use_scaler': True})
		PARAMS.update({'grid_search': True})
		PARAMS.update({'OOD_approach': 'equality'})

	elif 'random_forest' ==  technique:
		PARAMS.update({'grid_search': True})
		PARAMS.update({'OOD_approach': 'equality'})

	elif 'ocsvm' == technique:
		PARAMS.update({'OOD_approach': 'outlier'})
	
	elif 'oob' in technique:
		PARAMS.update({'arr_n_clusters': [3]})
		PARAMS.update({'arr_n_components': [2]}) 
		PARAMS.update({'tau': [0.01]}) # 0.0001, 0.01, 0.35
		PARAMS.update({'OOD_approach': 'outside_of_box'})

	elif 'knn' == technique:
		PARAMS.update({'arr_n_clusters': [2]}) #, 3, 5, 10
		PARAMS.update({'use_scaler': True})
		PARAMS.update({'OOD_approach': 'equality'})
		
	elif 'hdbscan' == technique:
		PARAMS.update({'min_samples': [5, 10, 15]})  #min_samples 5, 10, 15
		PARAMS.update({'OOD_approach': 'equality'})

	elif 'odin' == technique:
		PARAMS.update({'noiseMagnitude': 0.0025}) # gtsrb = 0.0025; cifar-10 = 0.0014
		PARAMS.update({'temper': 1000})
		# it is the rouding value of the min confidence threshold rounded in 4 decimals (0.10069 = 0.1007) 
		PARAMS.update({'threshold': 0.0237}) # gtsrb = 0.0237; cifar-10 = 0.1007
		PARAMS.update({'OOD_approach': 'temperature'})
	
	return PARAMS


def get_experiment_params(setting_id):
	'''
	-- id_dataset_name:
	'MNIST', 'GTSRB', 'BTSC', 'CIFAR-10'

	-- num_classes_to_monitor_ID:
	10, 43, 62, 10
	
	-- ood_dataset_name:
	'BTSC', 'GTSRB', 'CIFAR-10'

	-- num_classes_to_monitor_OOD:
	62, 43, 10
	
	-- modifications:
	('gtsrb', 'btsc'), ('cifar10', 'gtsrb'), ('gtsrb', 'cifar10')
	
	-- backend:
	keras = gtsrb, mnist; tensorflow = cifar10; pytorch = odin
	
	-- model_names:
	'leNet', 'vgg16'

	-- technique_names: 
	'baseline', 'knn', 'ocsvm', 'random_forest', 'sgd', 'hdbscan', 
		'oob', 'oob_isomap', 'oob_pca', 'oob_pca_isomap', odin
	'''

	PARAMS = {}

	# directory of datasets
	#root_dir = os.path.join('D:','\\backup_desktop_14-10-2020','GITHUB', 'phd_data_generation', 'data', 'modified')
	root_dir = os.path.join('C:', '\\Users', 'rsenaferre', 'Desktop', 'GITHUB', 'phd_data_generation', 'data', 'benchmark_dataset')
	PARAMS.update({'root_dir': root_dir})

	if setting_id == 1:
		# distributional shift (OOB) = GTSRB 

		id_dataset_name = 'GTSRB'
		PARAMS.update({'num_classes_to_monitor_ID': 43})
		PARAMS.update({'id_dataset_name': id_dataset_name})
		datasets = [id_dataset_name] 

		PARAMS.update({'ood_dataset_name': 'GTSRB'})
		PARAMS.update({'num_classes_to_monitor_OOD': 43})
		
		PARAMS.update({'data': ['brightness_severity_1', 'contrast_severity_1', 'defocus_blur_severity_1',
			'brightness_severity_5', 'contrast_severity_5', 'defocus_blur_severity_5',
			'elastic_transform_severity_1', 'gaussian_blur_severity_1', 'glass_blur_severity_1',
			'elastic_transform_severity_5', 'gaussian_blur_severity_5', 'glass_blur_severity_5',
			'saturate_severity_1', 'zoom_blur_severity_1', 'saturate_severity_5', 'zoom_blur_severity_5']})
		
		PARAMS.update({'backend': 'keras'}) 

		PARAMS.update({'model_names': 'leNet'})

		PARAMS.update({'technique_names': ['oob', 'oob_isomap', 'oob_pca']}) 

	elif setting_id == 2:
		# distributional shift (OOB) = CIFAR-10 

		id_dataset_name = 'CIFAR-10'
		PARAMS.update({'num_classes_to_monitor_ID': 10})
		PARAMS.update({'id_dataset_name': id_dataset_name})
		datasets = [id_dataset_name] 

		PARAMS.update({'ood_dataset_name': 'CIFAR-10'})
		PARAMS.update({'num_classes_to_monitor_OOD': 10})
		
		PARAMS.update({'data': ['brightness_severity_1', 'contrast_severity_1', 'defocus_blur_severity_1',
			'brightness_severity_5', 'contrast_severity_5', 'defocus_blur_severity_5',
			'elastic_transform_severity_1', 'gaussian_blur_severity_1', 'glass_blur_severity_1',
			'elastic_transform_severity_5', 'gaussian_blur_severity_5', 'glass_blur_severity_5',
			'saturate_severity_1', 'zoom_blur_severity_1', 'saturate_severity_5', 'zoom_blur_severity_5']})
		
		PARAMS.update({'backend': 'tensorflow'}) 

		PARAMS.update({'model_names': ['leNet']})

		PARAMS.update({'technique_names': ['oob', 'oob_isomap', 'oob_pca']}) 

	elif setting_id == 3:
		# distributional shift (ODIN) = GTSRB 

		id_dataset_name = 'GTSRB'
		PARAMS.update({'num_classes_to_monitor_ID': 43})
		PARAMS.update({'id_dataset_name': id_dataset_name})
		datasets = [id_dataset_name] 

		PARAMS.update({'ood_dataset_name': 'GTSRB'})
		PARAMS.update({'num_classes_to_monitor_OOD': 43})
		
		PARAMS.update({'data': ['brightness_severity_1', 'contrast_severity_1', 'defocus_blur_severity_1',
			'brightness_severity_5', 'contrast_severity_5', 'defocus_blur_severity_5',
			'elastic_transform_severity_1', 'gaussian_blur_severity_1', 'glass_blur_severity_1',
			'elastic_transform_severity_5', 'gaussian_blur_severity_5', 'glass_blur_severity_5',
			'saturate_severity_1', 'zoom_blur_severity_1', 'saturate_severity_5', 'zoom_blur_severity_5']})
		
		PARAMS.update({'backend': 'pytorch'}) 

		PARAMS.update({'model_names': 'leNet'})

		PARAMS.update({'technique_names': ['odin']}) 

	elif setting_id == 4:
		# distributional shift (ODIN) = CIFAR-10 

		id_dataset_name = 'CIFAR-10'
		PARAMS.update({'num_classes_to_monitor_ID': 10})
		PARAMS.update({'id_dataset_name': id_dataset_name})
		datasets = [id_dataset_name] 

		PARAMS.update({'ood_dataset_name': 'CIFAR-10'})
		PARAMS.update({'num_classes_to_monitor_OOD': 10})
		
		PARAMS.update({'data': ['brightness_severity_1', 'contrast_severity_1', 'defocus_blur_severity_1',
			'brightness_severity_5', 'contrast_severity_5', 'defocus_blur_severity_5',
			'elastic_transform_severity_1', 'gaussian_blur_severity_1', 'glass_blur_severity_1',
			'elastic_transform_severity_5', 'gaussian_blur_severity_5', 'glass_blur_severity_5',
			'saturate_severity_1', 'zoom_blur_severity_1', 'saturate_severity_5', 'zoom_blur_severity_5']})
		
		PARAMS.update({'backend': 'pytorch'}) 

		PARAMS.update({'model_names': ['leNet']})

		PARAMS.update({'technique_names': ['odin']}) 

	return PARAMS




def get_monitor_params(setting_id):

	PARAMS = {}

	if setting_id == 1:
		#for oob variations
		PARAMS.update({'is_build_monitors_by_class': True}) #True just for OOB-based monitors
		PARAMS.update({'arr_n_components': 2}) # 2, 3, 5, 10
		#for oob variations and knn
		PARAMS.update({'arr_n_clusters': 3}) # 2, 3, 5, 10
		#for ocsvm
		PARAMS.update({'min_samples': [5, 10, 15]})
		#for random forest and linear classifiers
		PARAMS.update({'use_grid_search': False})
		#for knn and sgd classifiers
		PARAMS.update({'use_scaler': False}) 
		#all methods
		PARAMS.update({'use_alternative_monitor': False}) # True = label -> act func -> save in the monitor; False = label -> act func if label == predicted -> save in the monitor
		PARAMS.update({'technique_names': ['oob', 'oob_isomap', 'oob_pca']}) #'baseline', 'knn', 'random_forest', 'sgd', 'ocsvm', 'oob', 'oob_isomap', 'oob_pca', 'oob_pca_isomap'
		PARAMS.update({'backend': 'keras'})
		PARAMS.update({'model_names': ['leNet']}) # 'leNet', 'vgg16', 'resnet' 

	elif setting_id == 2:
		PARAMS.update({'is_build_monitors_by_class': False})
		PARAMS.update({'threshold': None})
		PARAMS.update({'use_grid_search': False})
		PARAMS.update({'use_scaler': False}) 

		PARAMS.update({'magnitude': [0.0014, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]}) # based on paper 'generalized odin'
		PARAMS.update({'temperature': 1000})
		PARAMS.update({'use_alternative_monitor': False})
		PARAMS.update({'technique_names': ['odin']})
		PARAMS.update({'backend': 'pytorch'}) # for ODIN, using pytorch for now
		PARAMS.update({'model_names': ['leNet']})
		PARAMS.update({'use_gpu': True})

	return PARAMS


# reading from the desc.txt in data folder in the future
def get_data_params(setting_id):
	PARAMS = {}
	# directory of datasets
	#root_dir = os.path.join('D:','\\backup_desktop_14-10-2020','GITHUB', 'phd_data_generation', 'data', 'modified')
	root_dir = os.path.join('C:', '\\Users', 'rsenaferre', 'Desktop', 'GITHUB', 'phd_data_generation', 'data', 'training_set')
	PARAMS.update({'dataset_folder': root_dir})
	#PARAMS.update({'dataset_names': ['GTSRB']}) # 'MNIST', 'GTSRB', 'CIFAR-10'
	#PARAMS.update({'num_classes_to_monitor': [43]}) # 10, 43
	PARAMS.update({'validation_size': 0.3})

	return PARAMS