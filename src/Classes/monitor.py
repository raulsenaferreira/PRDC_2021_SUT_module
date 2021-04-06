

class Monitor:
	"""docstring for Monitor"""
	def __init__(self, monitor_name, layer_index=-2):
		super(Monitor, self).__init__()
		self.monitor_dataset = ''
		self.monitor_name = monitor_name
		self.class_to_monitor = 0
		self.layer_name = '' #optional
		self.layer_index = layer_index
		self.method = ''
		self.trainer = None
		self.dim_reduc_method = None
		self.monitors_folder = ''
		self.arrWeights = []