import numpy as np
from utils import klimaps
import random
from utils import normalize_dict
from utils import init_dictionary
from utils import klimaps_matrix

class Dictionary(object):
	"""Class to handle dictionaries"""
	def __init__(self):
		self.info = 'Class for dictionary storing'

	def dict_learn(self, Y, N, n_patch, max_iter, sparsity=6, seed=1, sparse_coder='klimaps', dict_learner='LS', init_method='data'):
		'''Learn dictionaries

		Y: gallery images (features pjected in LDA space)
		N: number of subjects
		n_patch: number of patches
		max_iter: maximum number of iterations for dictionary learning
		sparsity: sparsity level (if a list of values is pecified, one dict for each sparsity level will be learned)
		seed: for random number generation
		sparse_coder: "klimaps"
		dict_learner: "LS" (Least Squares)
		n_dicts: number of dictionaries
		init_method: initialization for the dictionaries; "data" or 'rand' 
		'''
		vectorized = True
			
		k = sparsity

		if init_method == 'data':
			D = init_dictionary(Y=Y, k=k, n_subj=N, n_patch=n_patch, method='data', seed=seed)
		elif init_method == 'random':
			D = init_dictionary(Y=Y, k=k, n_subj=N, n_patch=n_patch, method='rand', seed=seed)
		else:
			raise ValueError('Unrecognized Dictionary Initialization Method!!')

		D = normalize_dict(D)
		Dinv = np.linalg.pinv(D)

		non_zero = np.zeros([max_iter,1])
		residual = np.zeros([max_iter,1])
		for i in range(max_iter):

			print('\nIteration: ' + str(i+1) + ' ------------------------------------------------')
			print('\nComputing Sparse Codes with ' + sparse_coder + '...')

			if vectorized: 				
				X = klimaps_matrix(Y, D, Dinv, k, 1)
			else:			
				for p in range(Y.shape[1]):
					y = Y[:,p]		
					x = klimaps(y, D, Dinv, k, 1)					
					if p==0:
						X = np.expand_dims(x, axis=1)
					else:
						X = np.append(X,np.expand_dims(x, axis=1), axis=1)

			print('...Done')

			print('\nComputing Dictionary with ' + dict_learner + ' method...')

			l0 = np.zeros([N,1])
			Dict = []
			for j in range(N):			#for each sbj (gallery img)
				y_start = j*n_patch
				y_end = j*n_patch + n_patch
				Yj = Y[:,y_start:y_end]

				x_start_row = j*k
				x_end_row = j*k + k
				x_start_col = j*n_patch
				x_end_col = j*n_patch + n_patch
				Xj = X[x_start_row:x_end_row, x_start_col:x_end_col]

				X_sign = X[x_start_row:x_end_row, x_end_col:]
			
				if dict_learner == 'LS':
					Dj = np.matmul(Yj,np.linalg.pinv(Xj))
				else:
					raise ValueError('Unrecognized dictionary learner!!')

				Dict.append(Dj)
			
			print('...Done')

			D = np.squeeze(np.concatenate(Dict, axis=1))
			D = normalize_dict(D)
			Dinv = np.linalg.pinv(D)

			residual[i] = np.linalg.norm(np.dot(D,X) - Y)
			print('\nResidual: ' + str(residual[i]))
		
		self.learned_dict = D
		self.inv_dict = Dinv
		self.k = sparsity
