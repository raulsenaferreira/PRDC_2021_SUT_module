def get_monitor_params(technique):

	PARAMS = {}

	if technique in 'oob':
		#for oob variations
		PARAMS.update({'is_build_monitors_by_class': True}) #True just for OOB-based monitors
		PARAMS.update({'arr_n_components': 2}) # 2, 3, 5, 10
		#for oob variations and knn
		
		#for ocsvm
		PARAMS.update({'min_samples': [5, 10, 15]})
		#for random forest and linear classifiers
		PARAMS.update({'use_grid_search': False})
		#for knn and sgd classifiers
		PARAMS.update({'use_scaler': False}) 
		#all methods
		PARAMS.update({'use_alternative_monitor': False}) # True = label -> act func -> save in the monitor; False = label -> act func if label == predicted -> save in the monitor
		
		PARAMS.update({'model_names': ['leNet']}) # 'leNet', 'vgg16', 'resnet' 

	elif technique == 'odin':
		# for odin
		PARAMS.update({'is_build_monitors_by_class': False})
		PARAMS.update({'threshold': None})
		PARAMS.update({'use_grid_search': False})
		PARAMS.update({'use_scaler': False}) 
		PARAMS.update({'temperature': 1000})
		PARAMS.update({'use_alternative_monitor': False})
		PARAMS.update({'technique_names': ['odin']})
		PARAMS.update({'backend': 'pytorch'}) # for ODIN, using pytorch for now
		PARAMS.update({'model_names': ['leNet']})
		PARAMS.update({'use_gpu': True})

		if setting_id == 1:
			# gtsrb
			PARAMS.update({'magnitude': [0.0014, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]}) # based on paper 'generalized odin'
		elif setting_id == 2:
			# cifar-10
			PARAMS.update({'magnitude': [0.0014, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]}) # based on paper 'generalized odin'

	return PARAMS