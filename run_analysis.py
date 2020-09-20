import os
import run_ND_analysis as nd
from src.utils import util
import argparse


sep = util.get_separator()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("experiment_type_arg", help="Type of experiment (ID or OOD)")
	
	parser.add_argument("sub_field_arg", help="Type of ML problem (novelty_detection, distributional_shift,\
	 anomaly_detection, adversarial_attack)")

	parser.add_argument("save_experiments", type=int, help="Save experiments (1 for True or 0 for False)")

	parser.add_argument("parallel_execution", type=int, help="Parallelize experiments up to the number of physical \
		cores in the machine (1 for True or 0 for False)")

	args = parser.parse_args()

	text_parallel = 'serial'

	if args.parallel_execution:
		text_parallel = 'parallel'

	print('Starting {} processing of {} experiments on {} data...'.format(text_parallel,
	 args.sub_field_arg, args.experiment_type_arg))
	
	if args.save_experiments:
		print('The results will be saved on Neptune')
	else:
		print('The results will not be saved')
	
	repetitions = 1
	percentage_of_data = 0.1 #e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data

	if args.sub_field_arg == 'novelty_detection':
		nd.start(args.experiment_type_arg, args.save_experiments, args.parallel_execution, repetitions, percentage_of_data)
	elif args.sub_field_arg == 'distributional_shift':
		pass