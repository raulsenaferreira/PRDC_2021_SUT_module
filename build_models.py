from src.utils import util
from pathos.multiprocessing import ProcessingPool as Pool
from src import model_config as config
from src.Classes.dataset import Dataset
from src.Classes.model_builder import ModelBuilder


if __name__ == "__main__":
	dataset_names = ['MNIST', 'GTSRB']
	models_pool = []
	timeout = 1000

	#general settings
	sep = util.get_separator()
	validation_size = config.load_var_dict('validation_size')
	parallel_execution = True

	#datasets
	mnistObj = Dataset(dataset_names[0])
	mnistObj.validation_size = validation_size
	gtsrbObj = Dataset(dataset_names[1])
	gtsrbObj.validation_size = validation_size

	#Model 1: CNN with MNIST dataset
	model = config.load_model_settings(1)
	model.dataset = mnistObj
	#models_pool.append(model) 
	#Model 2: LeNet with GTRSB dataset
	model = config.load_model_settings(2)
	model.dataset = gtsrbObj
	models_pool.append(model) 
	#Model 3: Ensemble of CNN with MNIST dataset
	#model = load_model_settings(3)
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

		for model in models_pool:
			processes_pool.append(pool.apipe(model.runner.run, model)) 
		
		for process in processes_pool:
			history = process.get(timeout=timeout)
			

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