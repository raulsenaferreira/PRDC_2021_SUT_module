from src.utils import util
from src import model_config as model_cfg
from src.Classes.model_builder import ModelBuilder
from src.Classes.monitor import Monitor
from src.novelty_detection import config as config_ND
from pathos.multiprocessing import ProcessingPool as Pool
from src.Classes.dataset import Dataset
from src.novelty_detection.utils import abstraction_box
from src.Classes import act_func_based_monitor
from keras.models import load_model
from sklearn import manifold


if __name__ == "__main__":
	#general settings
	sep = util.get_separator()
	parallel_execution = False
	timeout = 1000

	experiment_type = 'novelty_detection'
	dataset_names = config_ND.load_vars(experiment_type, 'dataset_names')
	validation_size = config_ND.load_vars(experiment_type, 'validation_size')
	model_names = config_ND.load_vars(experiment_type, 'model_names')
	monitor_names = config_ND.load_vars(experiment_type, 'monitor_names')
	classes_to_monitor = config_ND.load_vars(experiment_type, 'classes_to_monitor')
	n_clusters_oob = 3 #fixo por enquanto, tranformar em array depois
	n_components_isomap = 2 #fixo por enquanto, tranformar em array depois
	
	for model_name, dataset_name, class_to_monitor in zip(model_names, dataset_names, classes_to_monitor):
		monitors = []
		# loading dataset
		dataset = Dataset(dataset_name)
		dataset.validation_size = validation_size

		# loading model
		model = ModelBuilder()
		#print(model.models_folder+model_name+'_'+dataset_name+'.h5')
		model = load_model(model.models_folder+model_name+'_'+dataset_name+'.h5')

		#Building monitors for Novelty Detection
		for monitor_name in monitor_names:
			monitor = None
			monitor_filename_prefix = model_name+'_'+dataset_name+'_'+monitor_name
			monitor = Monitor(monitor_name, experiment_type, monitor_filename_prefix)
			monitor.class_to_monitor = class_to_monitor

			if 'oob' in monitor_name:
				monitor.trainer = act_func_based_monitor
				monitor.method = abstraction_box.make_abstraction
				monitor.n_clusters = n_clusters_oob
				monitor.dim_reduc_method = None
			if 'isomap' in monitor_name:
				monitor.dim_reduc_method = manifold.Isomap(n_components = n_components_isomap)
				monitor.dim_reduc_filename_prefix = 'isomap_' + str(n_components_isomap) + '_components_' + dataset_name + '_class_' + str(monitor.class_to_monitor) + '.p'

			monitors.append(monitor)

		#Parallelizing the experiments (optional): one experiment per Core
		if parallel_execution:
			pool = Pool()
			processes_pool = []

			#pool.apipe(monitor.trainer.run, monitor, model)
			#process.get(timeout=timeout)

			for monitor in monitors:
				processes_pool.append(pool.apipe(monitor.trainer.run, monitor, model, dataset)) 
			
			for process in processes_pool:
				trained_monitor = process.get(timeout=timeout)
		else:
			for monitor in monitors:
				trained_monitor = monitor.trainer.run(monitor, model, dataset)
	#K = 3
	#monitoring one class in the GTRSB dataset using outside of box
	#monitor_name = "monitor_Box_GTRSB.p"
	#model_name = 'DNN_GTRSB.h5'
	#success = DNN_outOfBox_GTRSB_monitor.run(
	#	classToMonitor, config_ND.load_var_dict('monitors_folder'), monitor_name, models_folder, model_name, layer_name, validation_size, K, sep)

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

#monitoring one class in the GTRSB dataset using outside of box and dimensionality reduction
#model_name = 'DNN_GTRSB.h5'
#dim_reduc_method = 'isomap'
#monitor_name = "monitor_Box_"+dim_reduc_method+"_GTRSB.p"
#success = DNN_outOfBox_dimReduc_monitor.run(classToMonitor, monitors_folder, monitor_name, models_folder, model_name, layer_name, validation_size, K, dim_reduc_method, sep)

#monitoring one class in the MNIST dataset using outside of box and dimensionality reduction
#model_name = 'DNN_MNIST.h5'
#dim_reduc_method = 'isomap'
#monitor_name = "monitor_Box_"+dim_reduc_method+"_MNIST.p"
#success = DNN_outOfBox_dimReduc_MNIST_monitor.run(classToMonitor, monitors_folder, monitor_name, models_folder, model_name, layer_name, validation_size, K, dim_reduc_method, sep)

'''
#monitoring ensemble of CNNs in the MNIST using outside of box
monitors_ensemble_folder = monitors_folder+"outOfBox_ensembleDNN"+sep
monitor_ensemble_prefix = "monitor_Box_DNN_MNIST"
model_ensemble_prefix = 'DNN_ensemble_MNIST_'
num_cnn = 3
DNN_ensemble_outOfBox_MNIST_monitor.run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, validation_size, K)
'''