from dictionary import Dictionary
import numpy as np
from sparse_code import Sparse_code
import pickle

def evaluate_model(dataset, predictions, dictionary):
	'''Gives the classification accuracy'''

	subIDX = np.unique(dataset.labels_gallery)
	n_sbj = len(subIDX)
	n_patches = dataset.n_patch

	correct = 0
	labels = dataset.labels_test
	n_imgs = len(predictions)

	for i in range(n_imgs):
		if i==0:
			lbls = labels[i*n_patches]
		else:
			lbls = np.append(lbls, labels[i*n_patches])
	
	for i in range(len(predictions)):
		true_label = lbls[i].astype(int)
		ID = predictions[i][0].astype(int)

		if true_label == ID:
			correct = correct + 1

	accuracy = float(correct)/len(predictions)
	
	return accuracy*100


# --------------------------------------- MAIN Function ----------------------------------------------------------------------

def main():

	data_path = 'data/lfw_158_sbj.pkl'
	dict_path = 'data/lfw_158_sbj_dictionary.pkl'

	print('\nLoading data from disk... ')
	with open(data_path, 'rb') as input_file:
		dataset = pickle.load(input_file)

	info = dataset.get_data_info()	    
	print(info)
	print('Number of sbjs considered: ' + str(dataset.num_sbj))

	# Dictionary Learning --------------------------------------------------------------------------------------------------
	
	print('\nBuilding Dictionaries...')
	dictionaries = Dictionary()

	dictionaries.dict_learn(Y=dataset.gallery_lda, N=dataset.num_sbj, n_patch=dataset.n_patch, max_iter=10, init_method='data', sparsity=6)	
	
	with open(dict_path, "wb") as output_file:
		print('\nSaving Dictionary to disk...')
		pickle.dump(dictionaries, output_file, protocol=pickle.HIGHEST_PROTOCOL)
		print('Done... Saved to: ' + dict_path)


	#Classification ----------------------------------------------------------------------------------------------

	print('\nClassifying...')
	sc = Sparse_code()
	IDs = sc.klimaps_classify_learned_dict(dataset, dictionaries)
	
	print('\nNumber of images: ' + str(len(IDs)))

	accuracy = evaluate_model(dataset, IDs, dictionaries)

	print('\nAccuracy: ' + str(accuracy) + '%')
	
if __name__ == "__main__":
    main()
