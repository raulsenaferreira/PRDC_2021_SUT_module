import os
import argparse
from pathos.multiprocessing import ProcessingPool as Pool
from src import model_config as model_cfg
from src.Classes.dataset import Dataset


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("architecture", help="Type of DNN (lenet, vgg16, resnet)")

	parser.add_argument("dataset", help="Choose between pre-defined datasets (mnist, gtsrb,\
	 btsc, cifar-10, cifar-100, imagenet, lsun)")

	parser.add_argument("backend", help="Choose the backend library between keras or pytorch")

	parser.add_argument("verbose", type=int, help="Print the processing progress (1 for True or 0 for False)")

	parser.add_argument("save", type=int, help="Save trained model (1 for True or 0 for False)")

	parser.add_argument("percentage_of_data", type=int, default=100, help="e.g.: 10 = testing with 10% of test data; 100 = testing with all test data")

	args = parser.parse_args()

	#general settings
	parallel_execution = False
	perc_of_data = args.percentage_of_data / 100
	timeout = 1000
	save = True if args.save == 1 else False

	#datasets
	root_dir = os.path.join('C:', '\\Users', 'rsenaferre', 'Desktop', 'GITHUB', 'phd_data_generation', 'data', 'training_set', args.dataset)
	datasetObj = Dataset()

	'''
	## Tensorflow / Keras models
	#Model 1: CNN with MNIST dataset
	model_builder = model_cfg.load_settings('lenet_mnist', 'keras')
	model_builder.dataset = mnistObj
	models_pool.append(model_builder) 
	'''
	#Model 2: LeNet with GTRSB dataset
	model_builder = model_cfg.load_settings(args.architecture, args.dataset, args.backend)
	model_builder.dataset = datasetObj.load_dataset(root_dir, mode='train')
	
	'''
	#Model 3: ResNet with CIFAR-10 dataset
	model_builder = model_cfg.load_settings('resnet_cifar10')
	model_builder.dataset = cifarObj
	models_pool.append(model_builder) 
	'''
	#Model 4: VGG16 with GTSRB dataset
	#model_builder = model_cfg.load_settings('vgg16_gtsrb')
	#model_builder.dataset = gtsrbObj
	#models_pool.append(model_builder)

	## Pytorch models
	#Model 1: lenet with CIFAR-10 dataset
	#model_builder = model_cfg.load_settings('lenet_gtsrb', 'pytorch')
	#model_builder.dataset = datasetObj
	#models_pool.append(model_builder)

	#Model 5: Ensemble of CNN with MNIST dataset
	#model = model_cfg.load_settings(3)
	#model.dataset = mnistObj
	#models_pool.append(model)
	#Model 4: Ensemble of LeNet with GTRSB dataset
	#model = load_model_settings(4)
	#model.dataset = gtsrbObj
	#models_pool.append(model)

	
	#Serial version for the experiments
	history = model_builder.runner.run(model_builder, perc_of_data, save) 
	'''
	timeout = 0

	#Model 1: CNN with MNIST dataset
	experiments_pool.append(pool.apply_async(DNN_MNIST_model.run, [
		validation_size, batch_size, models_folder, epochs, model_name]))
	timeout+=1000
 	
 	#Model 2: LeNet with GTRSB dataset
	experiments_pool.append(pool.apply_async(DNN_GTRSB_model.run, [
		height, width, channels, trainPath, validation_size, models_folder,
			model_name_2, is_classification, num_classes]))
	timeout+=1000

	#Model 3: Ensemble of CNN with MNIST dataset
	experiments_pool.append(pool.apply_async(DNN_ensemble_MNIST_model.run, [
		batch_size, models_folder, epochs, model_name_prefix]))
	timeout+=3500  
	
	#Model 4: Ensemble of LeNet with GTRSB dataset
	experiments_pool.append(pool.apply_async(DNN_ensemble_GTRSB_model.run, [
		validation_size, batch_size_2, models_folder, epochs_2, model_name_prefix_2, sep, script_path])) 
	timeout+=3500

	for experiment in experiments_pool:
		history = experiment.get(timeout=timeout)

else:
	#Model 1: CNN with MNIST dataset
	#history = DNN_MNIST_model.run(validation_size, batch_size, models_folder, epochs, model_name)

	#Model 2: LeNet with GTRSB dataset
	#DNN_GTRSB_model.run(height, width, channels, trainPath, validation_size, models_folder,
	#	model_name_2, is_classification, num_classes)

	#Model 3: Ensemble of CNN with MNIST dataset
	DNN_ensemble_MNIST_model.run(batch_size, models_folder, epochs, model_name_prefix)

	#Model 4: Ensemble of LeNet with GTRSB dataset
	#DNN_ensemble_GTRSB_model.run(validation_size, batch_size_2, models_folder, epochs_2, model_name_prefix_2, sep, script_path)

	#Model 5: ALOCC model with MNIST
	'''