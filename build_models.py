from src.utils import util
from src.GTRSB_experiments import DNN_GTRSB_model
from src.GTRSB_experiments import DNN_ensemble_GTRSB_model
from src.MNIST_experiments import DNN_MNIST_model
from src.MNIST_experiments import DNN_ensemble_MNIST_model
from multiprocessing import Pool


if __name__ == "__main__":
	parallel_execution = False
	experiments_pool = []

	#general settings
	sep = util.get_separator()
	models_folder = "src"+sep+"bin"+sep+"models"+sep
	validation_size = 0.3

	#German traffic sign dataset
	trainPath = 'data'+sep+'GTS_dataset'+sep+"kaggle"+sep+"Train"+sep
	num_classes = 43
	is_classification = True
	height = 28
	width = 28
	channels = 3
	#MNIST
	batch_size = 128
	epochs = 12
	#GTSRB
	epochs_2 = 10
	batch_size_2 = 32

	#model 1
	model_name = 'DNN_MNIST.h5'
	#Model 2
	model_name_2 = 'DNN_GTRSB.h5'
	#Model 3
	model_name_prefix = 'DNN_ensemble_MNIST_'
	#Model 4
	model_name_prefix_2 = 'DNN_ensemble_GTRSB_'

	if parallel_execution:
		#Parallelizing the experiments (optional): one experiment per Core
		pool = Pool()
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
		