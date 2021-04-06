import os

class ModelBuilder:
	"""docstring for ModelBuilder"""
	def __init__(self, model_name=None, models_folder=None):
		super(ModelBuilder, self).__init__()
		self.binary = None
		self.algorithm = None
		self.batch_size = 0
		self.epochs = 0
		self.model_name = model_name
		self.num_cnn = 0 #used for ensembles
		self.models_folder = models_folder
		self.validation_size = 0.3
		self.num_classes = 0

		if self.models_folder != None and self.model_name != None:
			self.models_folder = os.path.join(self.models_folder, self.model_name)