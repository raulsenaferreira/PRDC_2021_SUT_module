import os

# reading from the desc.txt in data folder in the future
def get_data_params(dataset_name):
	PARAMS = {}
	# directory of datasets
	#root_dir = os.path.join('D:','\\backup_desktop_14-10-2020','GITHUB', 'phd_data_generation', 'data', 'modified')
	#root_dir = os.path.join('C:', '\\Users', 'rsenaferre', 'Desktop', 'GITHUB', 'phd_data_generation', 'data', 'training_set')
	
	PARAMS.update({'dataset_folder': root_dir})
	PARAMS.update({'validation_size': 0.3})

	if dataset_name == 'gtsrb':
		PARAMS.update({'dataset_names': ['GTSRB']})
		PARAMS.update({'num_classes_to_monitor': [43]})

	elif dataset_name == 'cifar10':
		PARAMS.update({'dataset_names': ['CIFAR-10']})
		PARAMS.update({'num_classes_to_monitor': [10]})
	

	return PARAMS