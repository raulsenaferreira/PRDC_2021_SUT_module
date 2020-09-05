import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.cluster import KMeans
from src.utils import util
from matplotlib.path import Path
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
from scipy.spatial import distance


sep = util.get_separator()

def plot_diff_images(image, reference):
	fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))


	for i, img in enumerate((image, reference)):
		for c, c_color in enumerate(('red', 'green', 'blue')):
			img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
			axes[c, i].plot(bins, img_hist / img_hist.max())
			img_cdf, bins = exposure.cumulative_distribution(img[..., c])
			axes[c, i].plot(bins, img_cdf)
			axes[c, 0].set_ylabel(c_color)

	axes[0, 0].set_title('Source')
	axes[0, 1].set_title('Reference')
	plt.tight_layout()
	plt.show()
	plt.imshow(np.squeeze(image), cmap='gray')
	plt.show()
	plt.imshow(np.squeeze(reference), cmap='gray')
	plt.show()


def histograms(image, reference):
	histSize = 256
	histRange = (0, 256) # the upper boundary is exclusive
	accumulate = False

	res_b = []
	res_g = []
	res_r = []
	c = 0
	for img, ref in zip(image, reference):
		b_hist = cv.calcHist(img, [0], None, [histSize], histRange, accumulate=accumulate)
		g_hist = cv.calcHist(img, [1], None, [histSize], histRange, accumulate=accumulate)
		r_hist = cv.calcHist(img, [2], None, [histSize], histRange, accumulate=accumulate)

		b_hist2 = cv.calcHist(ref, [0], None, [histSize], histRange, accumulate=accumulate)
		g_hist2 = cv.calcHist(ref, [1], None, [histSize], histRange, accumulate=accumulate)
		r_hist2 = cv.calcHist(ref, [2], None, [histSize], histRange, accumulate=accumulate)

		hist_h = 100
		cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
		cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
		cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

		cv.normalize(b_hist2, b_hist2, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
		cv.normalize(g_hist2, g_hist2, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
		cv.normalize(r_hist2, r_hist2, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

		b = distance.chebyshev(b_hist, b_hist2)
		print(b)
		g = distance.chebyshev(g_hist, g_hist2)
		print(g)
		r = distance.chebyshev(r_hist, r_hist2)
		print(r)
		print("average:", (b+g+r)/3)

		d = cv.compareHist(b_hist, b_hist2, cv.HISTCMP_CORREL)
		print("corr:", d)
		d = cv.compareHist(g_hist, g_hist2, cv.HISTCMP_CORREL)
		print("corr:", d)
		d = cv.compareHist(r_hist, r_hist2, cv.HISTCMP_CORREL)
		print("corr:", d)

		d = cv.compareHist(b_hist, b_hist2, cv.HISTCMP_CHISQR)
		print("Chi-sqr:", d)
		d = cv.compareHist(g_hist, g_hist2, cv.HISTCMP_CHISQR)
		print("Chi-sqr:", d)
		d = cv.compareHist(r_hist, r_hist2, cv.HISTCMP_CHISQR)
		print("Chi-sqr:", d)
		
		d = cv.compareHist(b_hist, b_hist2, cv.HISTCMP_INTERSECT)
		print("Intersection:", d)
		d = cv.compareHist(g_hist, g_hist2, cv.HISTCMP_INTERSECT)
		print("Intersection:", d)
		d = cv.compareHist(r_hist, r_hist2, cv.HISTCMP_INTERSECT)
		print("Intersection:", d)

		if (b+g+r)/3 == 0.0:
			print("equal ?", c)
		c+=1
		#res_b.append(b_hist)
		#res_g.append(g_hist)
		#res_r.append(r_hist)
	'''
	b = distance.chebyshev(res_b[0], res_b[1])
	print(b)
	g = distance.chebyshev(res_g[0], res_g[1])
	print(g)
	r = distance.chebyshev(res_r[0], res_r[1])
	print(r)

	print("average:", (b+g+r)/3)
	'''

def compare_histograms(img, ref, centered):
	def get_center_pixels(arr, npix):
		slices = [slice(shape/2-npix,shape/2+npix) for shape in arr.shape]
		return arr[slices]

	if centered:
		img = get_center_pixels(img,28)
		ref = get_center_pixels(ref,28)

	histSize = 256
	histRange = (0, 256) # the upper boundary is exclusive
	accumulate = False

	b_hist = cv.calcHist(img, [0], None, [histSize], histRange, accumulate=accumulate)
	g_hist = cv.calcHist(img, [1], None, [histSize], histRange, accumulate=accumulate)
	r_hist = cv.calcHist(img, [2], None, [histSize], histRange, accumulate=accumulate)

	b_hist2 = cv.calcHist(ref, [0], None, [histSize], histRange, accumulate=accumulate)
	g_hist2 = cv.calcHist(ref, [1], None, [histSize], histRange, accumulate=accumulate)
	r_hist2 = cv.calcHist(ref, [2], None, [histSize], histRange, accumulate=accumulate)

	hist_h = 400
	cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
	cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
	cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

	cv.normalize(b_hist2, b_hist2, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
	cv.normalize(g_hist2, g_hist2, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
	cv.normalize(r_hist2, r_hist2, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

	b = distance.chebyshev(b_hist, b_hist2)
	print(b)
	g = distance.chebyshev(g_hist, g_hist2)
	print(g)
	r = distance.chebyshev(r_hist, r_hist2)
	print(r)
	print("average:", (b+g+r)/3)

	return round((b+g+r)/3, 2)



def template_matching(img, template):
	#img = cv.imread('messi5.jpg',0)
	img2 = img.copy()
	#template = cv.imread('template.jpg',0)
	w, h = 14,14#template.shape[::-1]
	# All the 6 methods for comparison in a list
	methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
				'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
	for meth in methods:
		img = img2.copy()
		method = eval(meth)
		# Apply template Matching
		res = cv.matchTemplate(img,template,method)
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
		# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
		if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
			top_left = min_loc
		else:
			top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)
		cv.rectangle(img,top_left, bottom_right, 255, 2)
		plt.subplot(121),plt.imshow(res,cmap = 'gray')
		plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(img,cmap = 'gray')
		plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
		plt.suptitle(meth)
		plt.show()