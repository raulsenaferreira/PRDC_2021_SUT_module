# phd_experiments
Experiments regarding my PhD

## Usage
1) python build_models.py

2) python build_monitors.py

3) python run_experiments.py novelty_detection 0 0 0 

## arguments

sub_field_arg", help="Type of ML problem (novelty_detection, distributional_shift, anomaly_detection, adversarial_attack)

save_experiments", type=int, help="Save experiments (1 for True 0 for False)

parallel_execution", type=int, help="Parallelize experiments up to the number of physical cores in the machine (1 for True or 0 for False)

verbose, type=int, help="Print the processing progress (1 for True or 0 for False)"

###optional
percentage_of_data, type=int, default=1, help="e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data")

repetitions, type=int, default=1, help="number of repetitions for each experiment")