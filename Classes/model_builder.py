import os

class ModelBuilder:
	"""docstring for ModelBuilder"""
	def __init__(self, model_name):
		super(ModelBuilder, self).__init__()
		self.binary = None
		self.algorithm = None
		self.batch_size = 0
		self.epochs = 0
		self.model_name = model_name
		self.num_cnn = 0 #used for ensembles
		self.models_folder = os.path.join("src","bin","models", self.model_name)
		self.validation_size = 0.3