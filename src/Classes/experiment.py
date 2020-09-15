from src.utils import util


class Experiment:
	"""docstring for Experiment"""
	def __init__(self, name):
		super(Experiment, self).__init__()
		self.sep = util.get_separator()
		self.name = name
		self.monitors = []
		self.models = []
		self.datasets = []
		self.tester = None
		self.experiment_type = 'ID' # ID for in distribution, OOD for out-of-distribution