from src.utils import util
from src.novelty_detection import config as config_ND

from src.GTRSB_experiments import DNN_outOfBox_GTRSB_monitor
from src.GTRSB_experiments import DNN_outOfBox_dimReduc_monitor
from src.MNIST_experiments import DNN_outOfBox_MNIST_monitor
from src.GTRSB_experiments import DNN_ensemble_outOfBox_GTRSB_monitor
#from src.MNIST_experiments import SCGAN_MNIST_monitor
from src.MNIST_experiments import DNN_outOfBox_dimReduc_MNIST_monitor
from src.MNIST_experiments import DNN_ensemble_outOfBox_MNIST_monitor


#general settings
sep = util.get_separator()
#datasets
mnist = 1
gtsrb = 2
cifar = 3

#Building monitors for Novelty Detection
validation_size = config_ND.load_var_dict('validation_size')
classToMonitor = config_ND.load_var_dict('classToMonitor')
layer_index = config_ND.load_var_dict('layerToMonitor')



K = 3


#monitoring one class in the GTRSB dataset using outside of box
monitor_name = "monitor_Box_GTRSB.p"
model_name = 'DNN_GTRSB.h5'
success = DNN_outOfBox_GTRSB_monitor.run(
	classToMonitor, config_ND.load_var_dict('monitors_folder'), monitor_name, models_folder, model_name, layer_name, validation_size, K, sep)

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


#monitoring ensemble of CNNs in the MNIST using outside of box
monitors_ensemble_folder = monitors_folder+"outOfBox_ensembleDNN"+sep
monitor_ensemble_prefix = "monitor_Box_DNN_MNIST"
model_ensemble_prefix = 'DNN_ensemble_MNIST_'
num_cnn = 3
DNN_ensemble_outOfBox_MNIST_monitor.run(classToMonitor, layer_index, models_folder, monitors_ensemble_folder, monitor_ensemble_prefix, model_ensemble_prefix, num_cnn, validation_size, K)