# Experiments for the paper: Benchmarking Safety Monitors for Image Classifiers with Machine Learning

## Datasets
The benchmark datasets applied to these experiments were generated with our data generation module, that can be found in another repository: https://github.com/raulsenaferreira/PRDC_2021_Data_profile_module


## Usage
1) python build_models.py lenet gtsrb keras 0 0 100

2) python build_monitors.py novelty_detection oob gtsrb 1 0 0 100

3) python run_experiments.py novelty_detection oob cifar10_gtsrb 1 0 0 100


## arguments for 'build_models'

"architecture", help="Type of DNN (lenet, vgg16, resnet)"

"dataset", help="Choose between pre-defined datasets (mnist, gtsrb, btsc, cifar-10, cifar-100, imagenet, lsun)"

"backend", help="Choose the backend library between keras or pytorch"

"verbose", type=int, help="Print the processing progress (1 for True or 0 for False)"

"save", type=int, help="Save trained model (1 for True or 0 for False)"

"percentage_of_data", type=int, default=100, help="e.g.: 10 = testing with 10% of test data; 100 = testing with all test data"


## arguments for 'build_monitors' and 'run_experiments'

sub_field_arg", help="Type of ML problem (novelty_detection, distributional_shift, anomaly_detection, adversarial_attack)

technique", help="Type of SM technique (oob, odin, alocc)"

"dataset", help="dataset to apply to the experiments (gtsrb, cifar10, imagenet). For novelty, put ID_OOD dataset. Ex: gtsrb_btsc"

save_experiments", type=int, help="Save experiments (1 for True 0 for False)

parallel_execution", type=int, help="Parallelize experiments up to the number of physical cores in the machine (1 for True or 0 for False)

verbose, type=int, help="Print the processing progress (1 for True or 0 for False)"

percentage_of_data, type=int, default=100, help="e.g.: 10 = testing with 10% of test data; 100 = testing with all test data")


## Evaluation and results visualization (optional)
A module applied to generate tables and visualize results can be found in another repository:
https://github.com/raulsenaferreira/PRDC_2021_Evaluation_module
