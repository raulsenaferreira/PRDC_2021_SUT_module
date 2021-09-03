class Readout:
	"""docstring for Readout class"""
	def __init__(self):
		super(Readout, self).__init__()
		# general
		self.name = ''
		self.total_time = 0
		self.total_memory = 0

		# ML
		self.arr_classification_pred = []
		self.arr_classification_true = []
		self.ML_time = 0

		# SM
		self.arr_detection_SM = []
		self.arr_detection_true = []
		self.arr_reaction_SM = []
		self.arr_reaction_true = []
		self.SM_time = 0