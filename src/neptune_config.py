import neptune

def neptune_init(threat):
	neptune_root = 'raulsenaferreira/'
	log = 'saving experiments in '
	threat = threat.replace('_', '-') # adapting for neptune usage

	if threat != 'novelty-detection' and threat != 'adversarial-attack' and threat != 'distributional-shift' and threat != 'anomaly-detection' and threat != 'noise':
		threat='PhD'

	path = neptune_root+'{}'.format(threat) # 'PhD'
	print(log+path)

	neptune.init(path)