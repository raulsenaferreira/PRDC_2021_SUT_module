class Readout:
	"""docstring for Readout class"""
	def __init__(self):
		super(Readout, self).__init__()
		self.name = ''
		self.avg_acc = 0
		self.avg_time = 0
		self.avg_memory = 0
		self.avg_F1 = 0
		self.avg_cf = []
		self.std_deviation = 0