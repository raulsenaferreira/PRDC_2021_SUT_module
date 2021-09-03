import os
from src.threats.novelty_detection import build_ND_monitors as nd
import argparse
from src import monitor_config
from src import data_config


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("sub_field_arg", help="Type of threats (novelty_detection, distributional_shift,\
	 anomaly_detection, adversarial_attack)")

	parser.add_argument("technique", help="Type of Monitor (oob, odin, alooc, ensemble)")

	parser.add_argument("dataset", help="Dataset to apply in the Monitor (gtsrb, cifar10, imagenet)")

	parser.add_argument("save_experiments", type=int, help="Save experiments (1 for True or 0 for False)")

	parser.add_argument("parallel_execution", type=int, help="Parallelize experiments up to the number of physical \
		cores in the machine (1 for True or 0 for False)")

	parser.add_argument("verbose", type=int, help="Print the processing progress (1 for True or 0 for False)")

	parser.add_argument("percentage_of_data", type=int, default=100, help="e.g.: 10 = testing with 10% of test data; 100 = testing with all test data")

	args = parser.parse_args()

	text_save = 'The results will be saved on Neptune.' if args.save_experiments else 'The results will not be saved.'
	text_parallel = 'parallel' if args.parallel_execution == 1 else 'serial'
	text_verbose = 'verbose' if args.verbose == 1 else 'silence'

	print('Starting {} processing of {} experiments using {}% of data in {} mode. \n{}'.format(text_parallel,
	 args.sub_field_arg, args.percentage_of_data, text_verbose, text_save))

	args.percentage_of_data /= 100
	monitors_folder_root_dir = os.path.join('src', 'threats', args.sub_field_arg, 'bin', 'monitors')

	if args.sub_field_arg == 'novelty_detection':

		MONITOR_PARAMS = monitor_config.get_monitor_params(args.technique)
		MONITOR_PARAMS.update({'verbose': args.verbose})

		if args.technique in 'oob':
			# variations
			MONITOR_PARAMS.update({'technique_names': ['oob', 'oob_isomap', 'oob_pca']}) #'baseline', 'knn', 'random_forest', 'sgd', 'ocsvm', 'oob', 'oob_isomap', 'oob_pca', 'oob_pca_isomap'
			
			if args.dataset == 'gtsrb':
				MONITOR_PARAMS.update({'backend': 'keras'})
				MONITOR_PARAMS.update({'arr_n_clusters': 17}) # optimal for gtsrb according to Elbow analysis
			elif args.dataset == 'cifar10':
				MONITOR_PARAMS.update({'backend': 'tensorflow'})
				MONITOR_PARAMS.update({'arr_n_clusters': 5}) # 5 up to 10 according to Elbow analysis = CIFAR-10
			else:
				# default
				MONITOR_PARAMS.update({'arr_n_clusters': 3}) # 3 = default

			# Uncomment this line if you want to test without using clustering		
			#MONITOR_PARAMS.update({'arr_n_clusters': 0}) 
		
		DATA_PARAMS = data_config.get_data_params(args.dataset)

		nd.start(args.sub_field_arg, DATA_PARAMS, MONITOR_PARAMS, args.save_experiments, args.parallel_execution, \
			monitors_folder_root_dir, args.percentage_of_data)

	elif args.sub_field_arg == 'distributional_shift':
		pass