import os
import argparse



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
		
	parser.add_argument("sub_field_arg", help="Type of ML threats (novelty_detection, distributional_shift,\
	 anomaly_detection, adversarial_attack, noise)")

	parser.add_argument("technique", help="Type of SM technique (oob, odin, alooc)")

	parser.add_argument("dataset", help="dataset to apply to the experiments (gtsrb, cifar10, imagenet). \
		For novelty, put ID_OOD dataset. Ex: gtsrb_btsc")

	parser.add_argument("save_experiments", type=int, help="Save experiments (1 for True or 0 for False)")

	parser.add_argument("parallel_execution", type=int, help="Parallelize experiments up to the number of physical \
		cores in the machine (1 for True or 0 for False)")

	parser.add_argument("verbose", type=int, help="Print the processing progress (1 for True or 0 for False)")

	parser.add_argument("percentage_of_data", type=int, default=100, help="e.g.: 10 = testing with 10% of test data; 100 = testing with all test data")

	#parser.add_argument("repetitions", type=int, default=1, help="number of repetitions for each experiment")

	args = parser.parse_args()

	text_save = 'The results will be saved on Neptune.' if args.save_experiments else 'The results will not be saved.'
	text_parallel = 'parallel' if args.parallel_execution else 'serial'
	text_verbose = 'verbose' if args.parallel_execution else 'silence'

	print('Starting {} processing of {} experiments using {}% of data in {} mode. \n{}'.format(text_parallel,
	 args.sub_field_arg, args.percentage_of_data, text_verbose, text_save))

	repetitions = 1
	args.percentage_of_data /= 100

	if args.sub_field_arg == 'novelty_detection':
		from src.threats.novelty_detection import run_ND_experiments as nd
		from src.threats.novelty_detection import config
		
		exp_params = config.get_experiment_params(args.technique, args.dataset)

		if args.technique in 'oob':
			# variations
			exp_params.update({'technique_names': ['oob', 'oob_isomap', 'oob_pca']}) #'oob', 'oob_isomap', 'oob_pca', 'baseline', 'knn', 'random_forest', 'sgd', 'ocsvm', 'oob', 'oob_isomap', 'oob_pca', 'oob_pca_isomap'
			id_dataset = args.dataset.split('_')[0]
			if args.dataset == 'gtsrb' or id_dataset == 'gtsrb':
				exp_params.update({'arr_n_clusters': [17]}) # optimal for gtsrb according to Elbow analysis
			elif args.dataset == 'cifar10' or id_dataset == 'cifar10':
				exp_params.update({'arr_n_clusters': [5]}) # 5 up to 10 according to Elbow analysis = CIFAR-10
			else:
				# default
				exp_params.update({'arr_n_clusters': [3]}) # 3 = default

			# Uncomment this line if you want to test without using clustering		
			#exp_params.update({'arr_n_clusters': [0]}) 

			exp_params.update({'tau': 1.35}) # 1.0001, 1.1, 1.35

		elif args.technique in 'alooc':
			exp_params.update({'technique_names': ['alooc']})
			exp_params.update({'backend': 'keras'}) # tensorflow 
			exp_params.update({'use_scaler': False}) 
			exp_params.update({'optimizer': 'adam'}) #'rmsprop'
			exp_params.update({'model_number': 200})
			exp_params.update({'OOD_approach': 'adversarial'})

			id_dataset = args.dataset.split('_')[0]

			if args.dataset == 'gtsrb' or id_dataset == 'gtsrb':
				exp_params.update({'input_height': 28})
				exp_params.update({'input_width': 28})
				exp_params.update({'output_height': 28})
				exp_params.update({'output_width': 28})

				if exp_params['optimizer'] == 'adam':
					exp_params.update({'threshold': [1.2760, 0.9688, 0.8843, 1.6697, 0.9360, 0.8213, 0.5393, 0.5865, 0.5424, 1.2860,
					 0.5448, 0.8232, 1.2681, 1.1037, 0.8882, 1.2836, 1.2183, 0.9183, 1.2626, 0.7797, 0.6136, 0.6405, 1.6033, 1.3710,
					  0.6520, 0.7530, 0.9843, 0.7158, 1.4655, 0.9098, 0.5667, 0.6625, 0.7999, 1.2607, 1.5458, 0.9995, 0.6269, 1.2104,
					   0.9702, 1.0761, 1.1923, 0.6481, 0.5908]})
				elif exp_params['optimizer'] == 'rmsprop':
					exp_params.update({'threshold': [0.7822, 0.9886, 1.2168, 0.8090, 0.6733, 0.5947, 0.4281, 0.6435, 0.5605, 0.9560,
					 0.5882, 0.7490, 0.9709, 1.0137, 0.9536, 0.9262, 0.7104, 0.7690, 1.1362, 0.5887, 0.3941, 0.9005, 0.5545, 1.0487,
					  0.5859, 0.8698, 0.7923, 0.8037, 0.7356, 0.6227, 0.7223, 0.5667, 0.7826, 1.3079, 0.6974, 1.0795, 0.7107, 0.9344,
					   0.9542, 0.6228, 0.6937, 0.8745, 0.4306]})

			elif args.dataset == 'cifar10' or id_dataset == 'cifar10':
				exp_params.update({'input_height': 32})
				exp_params.update({'input_width': 32})
				exp_params.update({'output_height': 32})
				exp_params.update({'output_width': 32})

				if exp_params['optimizer'] == 'adam':
					exp_params.update({'threshold': [0.5617, 0.6404, 0.6140, 0.5592, 0.6109, 0.5932, 0.6071, 0.6035, 0.6072, 0.5650]})
				elif exp_params['optimizer'] == 'rmsprop':
					exp_params.update({'threshold': [0.7146, 0.5456, 0.6001, 0.7023, 0.5940, 0.5648, 0.7431, 0.5611, 0.7170, 0.5483]})

		elif args.technique in 'odin':
			id_dataset = args.dataset.split('_')[0]

			if args.dataset == 'gtsrb' or id_dataset == 'gtsrb':
				exp_params.update({'noiseMagnitude': '0.0025'})
			elif args.dataset == 'cifar10' or id_dataset == 'cifar10':	
				exp_params.update({'noiseMagnitude': '0.0014'})
			
			exp_params.update({'temperature': '1000'})

		exp_params.update({'sub_field': args.sub_field_arg})
		
		nd.start(exp_params, args.save_experiments, args.parallel_execution, \
			args.verbose, repetitions, args.percentage_of_data)

	elif args.sub_field_arg == 'distributional_shift':
		from src.threats.distributional_shift import run_DF_experiments as df
		from src.threats.distributional_shift import config

		exp_setting_id = 5 # odin + lenet + cifar10 id + gtsrb ood

		exp_params = config.get_experiment_params(exp_setting_id)

		exp_params.update({'sub_field': args.sub_field_arg})
		df.start(exp_params, args.save_experiments, args.parallel_execution, \
			args.verbose, repetitions, args.percentage_of_data)

	elif args.sub_field_arg == 'anomaly_detection':
		from src.threats.distributional_shift import run_DF_experiments as df
		from src.threats.anomaly_detection import config

		exp_setting_id = 13 # oob + gtsrb

		exp_params = config.get_experiment_params(exp_setting_id)

		exp_params.update({'sub_field': args.sub_field_arg})
		df.start(exp_params, args.save_experiments, args.parallel_execution, \
			args.verbose, repetitions, args.percentage_of_data)