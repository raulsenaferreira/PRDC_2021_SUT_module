from src.utils import util


class ModelBuilder:
	"""docstring for ModelBuilder"""
	def __init__(self):
		super(ModelBuilder, self).__init__()
		self.sep = util.get_separator()
		self.exec = None
		self.algorithm = None
		self.runner = None
		self.batch_size = 0
		self.epochs = 0
		self.models_name = None
		self.num_cnn = 0 #used for ensembles