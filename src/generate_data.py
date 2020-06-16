import numpy as np


def anomaly(image, mode):
		if mode == 'pixel_trap':
			indices = np.random.choice(image.shape[0], 2, replace=False)
			image[indices] = 0

		elif mode == 'row_add_logic':
			ind = int(image.shape[0]/2)-2
			image[ind+1] = image[ind]
			image[ind+2] = image[ind]
			image[ind+3] = image[ind]
			image[ind+4] = image[ind]
		
		elif mode == 'shifted_pixel':
			max_shift = 5
			m,n = image.shape[0], image.shape[1]
			col_start = np.random.randint(0, max_shift, image.shape[0])
			idx = np.mod(col_start[:,None] + np.arange(n), n)
			image = image[np.arange(m)[:,None], idx]

		return image