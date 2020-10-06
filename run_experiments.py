import os
from src.threats.novelty_detection import run_ND_experiments as nd
import argparse



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	#parser.add_argument("experiment_type_arg", help="Type of experiment (ID or OOD)")
	
	parser.add_argument("sub_field_arg", help="Type of ML problem (novelty_detection, distributional_shift,\
	 anomaly_detection, adversarial_attack, noise)")

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
		nd.start(args.sub_field_arg, args.save_experiments, args.parallel_execution, \
			args.verbose, repetitions, args.percentage_of_data)

	elif args.sub_field_arg == 'distributional_shift':
		pass