from src.utils import util


class ModelBuilder:
	"""docstring for ModelBuilder"""
	def __init__(self):
		super(ModelBuilder, self).__init__()
		self.sep = util.get_separator()
		self.binary = None
		self.algorithm = None
		self.batch_size = 0
		self.epochs = 0
		self.model_name = None
		self.num_cnn = 0 #used for ensembles
		self.models_folder = "src"+self.sep+"bin"+self.sep+"models"+self.sep