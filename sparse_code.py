import numpy as np
from utils import klimaps
from scipy import stats

class Sparse_code(object): 
	"""docstring for Sparse_code"""
	def __init__(self):
		super(Sparse_code, self).__init__()

	def klimaps_classify_learned_dict(self, dataset, dictionaries, method='mode', verbose=True):
		''' Klimaps classification for the learned dictionary

		Computes Sparse codes (alpha) for all the patches of a subjects, given dictionaries and returns the IDs

			Arguments:
			dataset: dataset object
			dictionaries: an object of class Dictionary
			method: 'mode'
		'''

		Y_all = dataset.test_lda
		n_patches = dataset.n_patch

		n_imgs = int(Y_all.shape[1]/n_patches)

		errors_d = []

		W = dataset.P

		for i in range(n_imgs):
			if i==0:
				lbls = dataset.labels_test[i*n_patches]
			else:
				lbls = np.append(lbls, dataset.labels_test[i*n_patches])

		IDs = np.zeros([n_imgs,1])


		for s in range(n_imgs):		# for all images
			
			#consider all the patches for the current test image
			Y = Y_all[:, s*n_patches : s*n_patches + n_patches]
			
			k = dictionaries.k
			D = dictionaries.learned_dict
			Dinv = dictionaries.inv_dict

			for ns in range(dataset.num_sbj):
				for a in range(k):
					if a==0 and ns==0:
						labels_gallery = dataset.labels_gallery[ns*n_patches + a]
					else:
						labels_gallery = np.append(labels_gallery, dataset.labels_gallery[ns*n_patches + a])

			IDsupp = []
			for p in range(n_patches):		#for all patches

				y = Y[:,p]
				alpha = klimaps(y, D, Dinv, k, 1)
				alpha_bool = np.array(alpha, dtype=bool)
				IDsupp.append(labels_gallery[alpha_bool])
			
			IDsupp = np.squeeze(np.concatenate(IDsupp))

			if method == 'mode':
				ID = stats.mode(IDsupp).mode[0].astype(int)
				IDs[s] = ID

			if verbose:
				print('\nReal ID:      ' + str(int(lbls[s])) + ' Corresponding to: ' + dataset.subjects[lbls[s]][0])
				print('Recovered ID: ' + str(ID) + ' Corresponding to: ' + dataset.subjects[ID][0])

		return IDs
		