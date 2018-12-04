import os
import operator
import warnings
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from utils import squeeze_mat
import scipy.io as spio

class Dataset(object):
    """Class to handle the datasets"""
 
    def __init__(self, name=None, img_path=None, feats_path=None):
        # name of the dataset
        '''
        self.subjects[sbj][0] = sbj_name            --> string
        self.subjects[sbj][1] = num_images          --> int
        self.subjects[sbj][2] = path_to_feats       --> string
        self.subjects[sbj][3] = feats_names         --> list of string
        self.subjects[sbj][4] = data                --> list ("num_images" ndarrays of dim 4096x72) (optional)
        '''
        self.name = name
        self.img_path = img_path
        self.feats_path = feats_path
       
        subjects_names = os.listdir(self.feats_path)
        numSubjectsTot = len(subjects_names)
       
        #Number of subjects in the dataset
        self.n_sbj_tot = numSubjectsTot
        subjects = {}

        idx = 0
        for sbj in subjects_names:
            feat_names_temp = []
            path_to_feats = feats_path+sbj
            feats_names = os.listdir(path_to_feats)

            subjects[idx] = [sbj, len(feats_names), path_to_feats, feats_names]
            idx = idx + 1

        #{key: sbj_idx, value: [sbj_name, num_images, path_to_feats, feats_names]
        self.subjects = subjects


    def load_mat_files(self, idx, max_img_per_sbj):
        #Loads mat files for the given subject index
        available_imgs = self.subjects[idx][1]
        num_images_to_pick = min(available_imgs, max_img_per_sbj)
        num_images = self.subjects[idx][1]
        all_imgs_idx = list(range(num_images))
        random.shuffle(all_imgs_idx)

        selected_imgs = all_imgs_idx[:num_images_to_pick]
        
        data = []
        for i, image_idx in enumerate(selected_imgs):
            feats_name = self.subjects[idx][3][image_idx]  #feat name
            if i==0:
                print(feats_name)
                print('  Took ' + str(num_images_to_pick) + ' images...')
            mat = spio.loadmat(self.subjects[idx][2] + '/' + feats_name)
            mat = mat['DeepFeat36_mS_mC_2F']        #shape: (4,1) (four scales)
            plain_mat = squeeze_mat(mat, feats='LR')
            data.append(plain_mat)
            
        return data, num_images_to_pick
        

    def split_train_and_test(self, num_sbj=None, min_imgs_per_sbj=None, max_img_per_sbj=10, seed=1, AR_test_experiment=None, experiment_train=None, experiment_test=None):
        '''Creates Gallery and test sets
           Arguments:
           num_sbj: Number of subjects to select
           min_imgs_per_sbj: minimum number of images for each subject
           max_img_per_sbj: maximum number of images for each subject (in test)
           seed: for reproducibility
        '''

        assert max_img_per_sbj >= min_imgs_per_sbj, "\n max_img_per_sbj field must have a value greater or equal to min_imgs_per_sbj"
        assert min_imgs_per_sbj >= 1, "\n There must be at least one image per subject"

        random.seed(seed)

        gallery = []
        test = []
        labels_gallery = []
        labels_test = []

        sub_dict = {}
        n_patch = 72  

        for idx in range(self.n_sbj_tot):
            sub_dict[idx] = self.subjects[idx][1]

        sorted_dict = sorted(list(sub_dict.items()), key=operator.itemgetter(1), reverse=True)

        assert min_imgs_per_sbj <= sorted_dict[0][1], "\nThere are no subjects with at least " + str(min_imgs_per_sbj) + " images in the dataset!"

        i=0
        candidate_idx = []
        while i < len(sorted_dict) and sorted_dict[i][1] >= min_imgs_per_sbj:
            candidate_idx.append(sorted_dict[i][0])
            i = i + 1
            pass
        n_candidates = i

        n_candidates = min(n_candidates, num_sbj)
        
        for i in range(n_candidates):

            print('\nSubject number: ' + str(i))
            
            #train --------------------------------------------------------------------
            index = candidate_idx[i]
            all_imgs_sbj, num_available_imgs = self.load_mat_files(index, max_img_per_sbj)
            rand_idx = random.randint(0,num_available_imgs-1)

            gallery.append(all_imgs_sbj[rand_idx])
            labels_gallery.append(np.ones([1,n_patch])*index)

            #test ---------------------------------------------------------------------
            idx_test_set = list(range(num_available_imgs))
            idx_test_set.remove(rand_idx)   

            for j, t_idx in enumerate(idx_test_set):
                test.append(all_imgs_sbj[t_idx])
                labels_test.append(np.ones([1,n_patch])*index)

        gallery = np.squeeze(np.concatenate(gallery,axis=1))
        labels_gallery = np.squeeze(np.concatenate(labels_gallery,axis=1))
        test = np.squeeze(np.concatenate(test,axis=1))
        labels_test = np.squeeze(np.concatenate(labels_test,axis=1))

        print(gallery.shape)
        print(labels_gallery.shape)
        print(test.shape)
        print(labels_test.shape)

        self.n_patch = n_patch


    def project_data(self, projection_type='LDA', normalize=True, do_pca=True):

        print('\nApplying LDA...')
        if normalize:
            scaler = StandardScaler()
            scaler.fit(self.gallery.T)
            self.gallery = scaler.transform(self.gallery.T)
            self.test = scaler.transform(self.test.T)

        if do_pca:
            pca = PCA(.95)
            pca.fit(self.gallery)
            self.pca = pca
            gallery_pca = pca.transform(self.gallery)
            test_pca = pca.transform(self.test)
            pca_matrix = pca.components_

            print('PCA dimesion (95% of the variance retained) --> ' + str(gallery_pca.shape))

        if projection_type=='LDA':
            #projection = LDA(n_components=self.num_sbj-1, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)
            projection = LDA(n_components=self.num_sbj-1, shrinkage='auto', solver='eigen')
            projection.fit(gallery_pca, self.labels_gallery)
            self.gallery_lda = projection.transform(gallery_pca)
            self.test_lda = projection.transform(test_pca)
            self.projection = projection
            self.projection_type = projection_type
            lda_matrix = projection.scalings_

            self.gallery_lda = self.gallery_lda.T
            self.test_lda = self.test_lda.T

            print('LDA dimension --> ' + str(self.gallery_lda.shape))

        W = np.dot(lda_matrix.T,pca_matrix)
        self.P = W

    def get_data_info(self):
        return ("\nDataset\n- name: %s\n- img_path: %s\n- feats_path: %s\n- total_number_of_sbjs: %s\n" % (self.name, self.img_path, self.feats_path, self.n_sbj_tot))