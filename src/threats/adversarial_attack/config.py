
def load_novelty_detection_experiment(experiment_number):
	#Experiment 1: DNN with outside-of-box monitor
	experiment = Experiment('DNN+OB')
	#models
	dnn_mnist = ModelBuilder()
	dnn_mnist.model_name = 'DNN_MNIST.h5'
	dnn_mnist.exec = dnn_oob_tester
	dnn_gtsrb = ModelBuilder()
	dnn_gtsrb.model_name = 'DNN_GTRSB.h5'
	dnn_gtsrb.exec = dnn_oob_tester
	modelsObj = [dnn_mnist, dnn_gtsrb]
	#monitors
	monitorObjMNIST = Monitor("monitor_Box_MNIST.p", 7, -2)
	monitorObjMNIST.method = abstraction_box.find_point
	monitorObjGTSRB = Monitor("monitor_Box_GTRSB.p", 7, -2)
	monitorObjGTSRB.method = abstraction_box.find_point
	monitorsObj = [monitorObjMNIST, monitorObjGTSRB]
	#building the class experiment 1
	experiment.datasets = datasetObjs
	experiment.models = modelsObj
	experiment.monitors = monitorsObj
	experiment.evaluator = dnn_oob_evaluator
	#adding to the pool of experiments
	experiments_pool.append(experiment)
	
	#Experiment 2: DNN with outside-of-box monitor using non-linear dimensionality reduction
	experiment = Experiment('DNN+OB+NL')
	#using the same ML models from the Experiment 1
	dnn_mnist = ModelBuilder()
	dnn_mnist.model_name = 'DNN_MNIST.h5'
	dnn_mnist.exec = dnn_oob_tester
	dnn_gtsrb = ModelBuilder()
	dnn_gtsrb.model_name = 'DNN_GTRSB.h5'
	dnn_gtsrb.exec = dnn_oob_tester
	modelsObj = [dnn_mnist, dnn_gtsrb]
	#monitors
	monitorObjMNIST = Monitor("monitor_Box_MNIST.p", 7, -2)
	monitorObjMNIST.method = abstraction_box.find_point
	monitorObjGTSRB = Monitor("monitor_Box_GTRSB.p", 7, -2)
	monitorObjGTSRB.method = abstraction_box.find_point
	monitorsObj = [monitorObjMNIST, monitorObjGTSRB]