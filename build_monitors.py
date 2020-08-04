from src.utils import util
from src import model_config as model_cfg
from src.Classes.model_builder import ModelBuilder
from src.novelty_detection import config as config_ND
from pathos.multiprocessing import ProcessingPool as Pool
from src.Classes.dataset import Dataset
from src.novelty_detection import nd_monitors
from keras.models import load_model


if __name__ == "__main__":
	#general settings
	sep = util.get_separator()
	parallel_execution = True
	timeout = 1000
	arr_n_components_isomap = [2, 3, 5, 10]
	arr_n_clusters_oob = [2, 3, 4, 5]

	experiment_type = 'novelty_detection'
	dataset_names = config_ND.load_vars(experiment_type, 'dataset_names')
	validation_size = config_ND.load_vars(experiment_type, 'validation_size')
	model_names = config_ND.load_vars(experiment_type, 'model_names')
	monitor_names = config_ND.load_vars(experiment_type, 'monitor_names')
	classes_to_monitor = config_ND.load_vars(experiment_type, 'classes_to_monitor')
	perc_of_data = 1 #e.g.: 0.1 = testing with 10% of test data; 1 = testing with all test data
	
	for model_name, dataset_name, class_to_monitor in zip(model_names, dataset_names, classes_to_monitor):
		# loading dataset
		dataset = Dataset(dataset_name)
		dataset.validation_size = validation_size

		# loading model
		model = ModelBuilder()
		model = load_model(model.models_folder+model_name+'_'+dataset_name+'.h5')

		#oob_ablation_study(n_components_isomap)
		for n_components_isomap in arr_n_components_isomap:

			#Building monitors for Novelty Detection
			monitors = nd_monitors.build_monitors(dataset_name, monitor_names, class_to_monitor,
			 n_components_isomap=n_components_isomap)

			#Parallelizing the experiments (optional): one experiment per Core
			if parallel_execution:
				pool = Pool()
				processes_pool = []

				for monitor in monitors:
					processes_pool.append(pool.apipe(monitor.trainer.run, monitor, model, dataset, perc_of_data)) 
				
				for process in processes_pool:
					trained_monitor = process.get(timeout=timeout)
			else:
				for monitor in monitors:
					trained_monitor = monitor.trainer.run(monitor, model, dataset, perc_of_data)
	
#monitoring ensemble of CNNs in the GTRSB using outside of box
#layer_index = 8
#monitors_ensemble_folder = monitors_folder+"outOfBox_ensembleDNN"+sep
#monitor_ensemble_prefix = "monitor_Box_DNN_"
#model_ensemble_prefix = 'DNN_ensemble_GTRSB_'
#num_cnn = 5
#success = DNN_ensemble_outOfBox_GTRSB_monitor.run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, validation_size, K, sep, script_path)

#monitoring one class in the MNIST dataset using outside of box
#monitor_name = "monitor_Box_MNIST.p"
#model_name = 'DNN_MNIST.h5'
#success = DNN_outOfBox_MNIST_monitor.run(classToMonitor, monitors_folder, monitor_name, models_folder, model_name, layer_name, validation_size, K)

#monitoring one classe in the MNIST dataset using a DCGAN:
#epochs=5
#batch_size=128
#sample_interval=500
#monitors_folder_checkpoint = monitors_folder+sep+'SCGAN_checkpoint'
#monitor_name = 'SCGAN_MNIST_'
#monitor = SCGAN_MNIST_monitor.ALOCC_Model(input_height=28,input_width=28)
#X_train, y_train, _, _, _ = util.load_mnist(onehotencoder=False)
#monitor.train(X_train, y_train, classToMonitor, epochs, batch_size, sample_interval, monitors_folder_checkpoint, monitor_name)


'''
#monitoring ensemble of CNNs in the MNIST using outside of box
monitors_ensemble_folder = monitors_folder+"outOfBox_ensembleDNN"+sep
monitor_ensemble_prefix = "monitor_Box_DNN_MNIST"
model_ensemble_prefix = 'DNN_ensemble_MNIST_'
num_cnn = 3
DNN_ensemble_outOfBox_MNIST_monitor.run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, validation_size, K)
'''