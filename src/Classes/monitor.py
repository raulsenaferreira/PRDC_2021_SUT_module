from src.utils import util


class Monitor:
	"""docstring for Monitor"""
	def __init__(self, type_of_monitoring, monitor_name, classToMonitor, layer_index):
		super(Monitor, self).__init__()
		self.sep = util.get_separator()
		self.monitor_name=monitor_name
		self.classToMonitor = classToMonitor
		self.layer_name = '' #optional
		self.layer_index = layer_index
		self.method = ''
		self.trainer = None
		self.dim_reduc_method = None
		self.monitors_folder = "src"+self.sep+type_of_monitoring+self.sep+"bin"+self.sep+"monitors"+self.sep