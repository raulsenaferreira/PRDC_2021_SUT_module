from src.utils import util


class Monitor:
	"""docstring for Monitor"""
	def __init__(self, monitor_name, classToMonitor, layer_index):
		super(Monitor, self).__init__()
		self.sep = util.get_separator()
		self.monitor_name=monitor_name
		self.classToMonitor = classToMonitor
		self.layer_name = '' #optional
		self.layer_index = layer_index
		self.method = ''
		self.monitors_folder="src"+self.sep+"bin"+self.sep+"monitors"+self.sep