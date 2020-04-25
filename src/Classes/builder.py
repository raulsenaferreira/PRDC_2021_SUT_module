from src.utils import util


class ModelBuilder:
	"""docstring for ModelBuilder"""
	def __init__(self):
		super(ModelBuilder, self).__init__()
		self.sep = util.get_separator()
		self.exec = None
		self.isTestOneClass = True
		self.models_folder = "src"+self.sep+"bin"+self.sep+"models"+self.sep
		self.models_name = None