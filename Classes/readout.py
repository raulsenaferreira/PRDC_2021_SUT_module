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

		self.arr_pos_neg_ID_pred, self.arr_pos_neg_ID_true = [], []
		self.arr_false_negative_ID = {}
		self.arr_true_negative_ID = {}
		self.arr_false_positive_ID = {}
		self.arr_true_positive_ID = {}

		self.arr_pos_neg_OOD_pred, self.arr_pos_neg_OOD_true = [], []
		self.arr_false_negative_OOD = {}
		self.arr_false_positive_OOD = {}
		self.arr_true_positive_OOD = {}
		self.arr_true_negative_OOD = {}