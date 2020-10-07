# phd_experiments
Experiments regarding my PhD

## Usage
1) python build_models.py lenet cifar-10

2) python build_monitors.py novelty_detection 0 0 0

3) python run_experiments.py novelty_detection 0 0 0 100

## arguments for 'build_models'



## arguments for 'build_monitors' and 'run_experiments'

sub_field_arg", help="Type of ML problem (novelty_detection, distributional_shift, anomaly_detection, adversarial_attack)

save_experiments", type=int, help="Save experiments (1 for True 0 for False)

parallel_execution", type=int, help="Parallelize experiments up to the number of physical cores in the machine (1 for True or 0 for False)

verbose, type=int, help="Print the processing progress (1 for True or 0 for False)"

percentage_of_data, type=int, default=100, help="e.g.: 10 = testing with 10% of test data; 100 = testing with all test data")

### optional

repetitions, type=int, default=1, help="number of repetitions for each experiment")