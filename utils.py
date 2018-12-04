import numpy as np

def klimaps(y, D, Dinv, k, max_iter):
	
	alpha = np.matmul(Dinv,y)
	a = np.sort(np.abs(alpha))
	lambda_ = 1/a[-(k+1)];
	
	for i in range(max_iter):
		# apply sparsity constraction mapping: increase sparsity
		b = 1-np.exp(-lambda_*np.abs(alpha))
		beta = np.multiply(alpha,b)
		#apply the orthogonal projection
		alpha = beta-np.matmul(Dinv,(np.matmul(D,beta)-y))
		#update the lambda coefficient
		a = np.sort(np.abs(alpha))
		lambda_ = 1/a[-(k+1)]

		if i != 0:
			if np.sum(np.abs(alpha - alpha_prev)) < 1e-2:
				break
		alpha_prev = alpha

	idx = np.argsort(np.abs(alpha))
	alpha[idx[0:-k]] = 0

	#Least Square
	non_zero = np.nonzero(alpha)[0]
	D = D[:,non_zero]
	a = np.matmul(np.linalg.pinv(D),y)
	alpha[non_zero] = a

	return alpha


def klimaps_matrix(Y, D, Dinv, k, max_iter):
	'''Vectorized version of k-limaps'''

	Alpha = np.matmul(Dinv,Y)
	m = D.shape[1]
	N = Alpha.shape[1]
	P = np.eye(m) - np.matmul(Dinv, D)
	a = -np.sort(-np.abs(Alpha), axis=0)

	Lambda = np.tile(np.divide(1,a[k,:]),(m,1))

	for i in range(max_iter):
		b = np.multiply(Alpha,np.exp(np.multiply(-Lambda,abs(Alpha))))
		Alpha = Alpha - np.matmul(P,b)
		a = -np.sort(-np.abs(Alpha), axis=0)
		Lambda = np.tile(np.divide(1,a[k,:]),(m,1))

	idx = np.argsort(-np.abs(Alpha), axis=0)
	for i in range(N):
		Alpha[idx[k:m,i],i] = 0
		Alpha[idx[0:k,i],i] = np.matmul(np.linalg.pinv(D[:,idx[0:k,i]]), Y[:,i])

	return Alpha


def normalize_dict(D):
	'''Normalize dict columns to unit norm'''
	n = np.linalg.norm(D, axis=0)
	D_norm = D / (n + np.finfo(float).eps)

	return D_norm

def squeeze_mat(mat, feats):
	ns = 4
	nc = 9
	nf = 2
	nshbl = 3
	nLR = 4
	featLen = 4096
	if feats == 'new':
		npatch = ns*nc*nf*nshbl
	elif feats == 'LR':
		npatch = ns*nc*nf*nLR
	else:
		npatch = ns*nc*nf
	squeezed = np.zeros([featLen,npatch])

	if feats == 'new':
		i = 0
		for shbl in range(nshbl):
			for s in range(ns):
				for c in range(nc):
					for f in range(nf):
						squeezed[:,i] = np.squeeze(mat[0][shbl][0][s][0][c][0][f])
						i = i + 1
	if feats == 'LR':
		i = 0
		for s in range(ns):
			for c in range(nc):
				for f in range(nf):
					for lr in range(nLR):
						a = np.squeeze(mat[0][s][0][c][0][f][0][lr])
						squeezed[:,i] = np.squeeze(mat[0][s][0][c][0][f][0][lr])
						i = i + 1
	else:
		i = 0
		for s in range(ns):
			for c in range(nc):
				for f in range(nf):
					squeezed[:,i] = np.squeeze(mat[0][s][0][c][0][f])
					i = i + 1
	return squeezed


def init_dictionary(Y=None, k=6, n_subj=None, n_patch=None, method='data', seed=1):
	''' Functon for dictionary initialization'''
	
	np.random.seed(seed)

	if method == 'data':
		print('\nInitializing Dictionary with data...')
		for j in range(n_subj):
			start = j*n_patch
			end = j*n_patch + n_patch
			Yj = Y[:,start:end]
			rand_vect = np.arange(n_patch)
			np.random.shuffle(rand_vect)
			k_rand = rand_vect[:k]
	
			if j==0:
				D = Yj[:,k_rand]
			else:
				D = np.append(D,Yj[:,k_rand], axis=1)
	
	elif method == 'rand':
		print('\nInitializing Dictionary with random numbers...')
		D = np.random.randn(Y.shape[0],k*n_subj)

	else:

		raise ValueError('Unrecognized method!!')

	return D
