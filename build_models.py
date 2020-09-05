from pathos.multiprocessing import ProcessingPool as Pool
from src import model_config as model_cfg
from src.Classes.dataset import Dataset


if __name__ == "__main__":
	perc_of_data = 0.1 #e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data

	dataset_names = ['MNIST', 'GTSRB', 'CIFAR-10']
	models_pool = []
	timeout = 1000

	#general settings
	parallel_execution = False
	
	#datasets
	#mnistObj = Dataset(dataset_names[0])
	
	gtsrbObj = Dataset(dataset_names[1])
	
	#cifarObj = Dataset(dataset_names[2])
	

	'''
	#Model 1: CNN with MNIST dataset
	model_builder = model_cfg.load_settings('lenet_mnist')
	model_builder.dataset = mnistObj
	models_pool.append(model_builder) 
	
	#Model 2: LeNet with GTRSB dataset
	model_builder = model_cfg.load_settings('lenet_gtsrb')
	model_builder.dataset = gtsrbObj
	models_pool.append(model_builder) 
	
	#Model 3: ResNet with CIFAR-10 dataset
	model_builder = model_cfg.load_settings('resnet_cifar10')
	model_builder.dataset = cifarObj
	models_pool.append(model_builder) 
	'''
	#Model 4: VGG16 with GTSRB dataset
	model_builder = model_cfg.load_settings('vgg16_gtsrb')
	model_builder.dataset = gtsrbObj
	models_pool.append(model_builder)

	#Model 5: Ensemble of CNN with MNIST dataset
	#model = model_cfg.load_settings(3)
	#model.dataset = mnistObj
	#models_pool.append(model)
	#Model 4: Ensemble of LeNet with GTRSB dataset
	#model = load_model_settings(4)
	#model.dataset = gtsrbObj
	#models_pool.append(model)

	if parallel_execution:
		#Parallelizing the experiments (optional): one experiment per Core
		pool = Pool()
		processes_pool = []
		timeout *= len(models_pool)

		for model_builder in models_pool:
			processes_pool.append(pool.apipe(model_builder.runner.run, model_builder, perc_of_data)) 
		
		for process in processes_pool:
			history = process.get(timeout=timeout)
			
	else:
		#Serial version for the experiments
		for model_builder in models_pool:
			history = model_builder.runner.run(model_builder, perc_of_data) 
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