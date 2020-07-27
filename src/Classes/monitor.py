from src.utils import util


class Monitor:
	"""docstring for Monitor"""
	def __init__(self, monitor_name, type_of_monitoring, monitor_filename_prefix, layer_index=-2):
		super(Monitor, self).__init__()
		self.sep = util.get_separator()
		self.monitor_dataset = ''
		self.monitor_name = monitor_name
		self.monitor_filename_prefix=monitor_filename_prefix
		self.class_to_monitor = 0
		self.layer_name = '' #optional
		self.layer_index = layer_index
		self.method = ''
		self.trainer = None
		self.dim_reduc_method = None
		self.monitors_folder = "src"+self.sep+type_of_monitoring+self.sep+"bin"+self.sep+"monitors"+self.sep