# phd_experiments
Experiments regarding my PhD

## Usage
1) python build_models.py

2) python build_monitors.py

3) python run_analysis.py OOD novelty_detection 0 0

## arguments
experiment_type_arg", help="Type of experiment (ID or OOD)

sub_field_arg", help="Type of ML problem (novelty_detection, distributional_shift, anomaly_detection, adversarial_attack)

save_experiments", type=int, help="Save experiments (1 for True or 0 for False)

parallel_execution", type=int, help="Parallelize experiments up to the number of physical cores in the machine (1 for True or 0 for False)