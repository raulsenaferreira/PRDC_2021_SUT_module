from src.utils import util


class Experiment:
	"""docstring for Experiment"""
	def __init__(self, acronym):
		super(Experiment, self).__init__()
		self.sep = util.get_separator()
		self.acronym = acronym
		self.monitors = []
		self.models = []
		self.datasets = []
		self.tester = None