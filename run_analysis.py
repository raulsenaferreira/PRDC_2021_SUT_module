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

	args = parser.parse_args()
	print('Starting {} experiments on {} data...'.format(args.sub_field_arg, args.experiment_type_arg))

	if args.sub_field_arg == 'novelty_detection':
		nd.start(args.experiment_type_arg)