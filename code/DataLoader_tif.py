import os
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils import data
from PIL import Image
import glob

def read_data_paths_form_dict():

    ct_data_path = r'D:\MRI_CT_data\Train\CT_train'
    mri_data_path = r'D:\MRI_CT_data\Train\MR_train'

    ct_data_path = os.path.normpath(ct_data_path)
    mri_data_path = os.path.normpath(mri_data_path)

    mri_list_dir = os.listdir(mri_data_path) #type  -> list
    ct_list_dir = os.listdir(ct_data_path)

    total_samples = len(mri_list_dir)
    my_dict = {}

    for i in range(total_samples):

        # print(i)
        mr_f_path = os.path.join(mri_data_path, mri_list_dir[i])
        ct_f_path = os.path.join(ct_data_path, ct_list_dir[i])

        my_dict[mr_f_path] = ct_f_path   #just doing indexing

    return my_dict


def read_data_paths_form_list():

    ct_data_path = r'D:\MRI_CT_data\Train\CT_train'
    mri_data_path = r'D:\MRI_CT_data\Train\MR_train'

    ct_data_path = os.path.normpath(ct_data_path)
    mri_data_path = os.path.normpath(mri_data_path)

    mri_list_dir = os.listdir(mri_data_path)
    ct_list_dir = os.listdir(ct_data_path)

    total_samples = len(mri_list_dir)
    x_list = []
    y_list = []

    for i in range(total_samples):

        # print(i)
        mr_f_path = os.path.join(mri_data_path, mri_list_dir[i])
        ct_f_path = os.path.join(ct_data_path, ct_list_dir[i])

        x_list.append(mr_f_path)
        y_list.append(ct_f_path)

    return x_list, y_list


def test_data_paths_form_list():

    ct_test_data_path = r'D:\MRI_CT_data\Test\CT_test'
    mr_test_data_path = r'D:\MRI_CT_data\Test\MR_test'

    ct_test_data_path = os.path.normpath(ct_test_data_path)
    mr_test_data_path = os.path.normpath(mr_test_data_path)

    ct_test_listdir = os.listdir(ct_test_data_path)
    mr_test_listdir = os.listdir(mr_test_data_path)

    test_samples = len(mr_test_listdir)

    x_test = []
    y_test = []

    for i in range(test_samples):

        mr_test_path = os.path.join(mr_test_data_path,mr_test_listdir[i])
        ct_test_path = os.path.join(ct_test_data_path,ct_test_listdir[i])

        x_test.append(mr_test_path)
        y_test.append(ct_test_path)

    return x_test, y_test


class Dataset(data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, dir_data):
        self.dir_data = dir_data
        # print(os.path.join(self.dir_data,'MR*','*.tif'))
        self.filespaths_mr=glob.glob(os.path.join(self.dir_data,'MR*','*.tif'))

    def __len__(self):
        'denotes the total number of samples'
        return len(self.filespaths_mr)

    def __getitem__(self, item):
        'Generates one sample of data'
        #select sample
        path_mr=self.filespaths_mr[item]
        filename=path_mr.split(os.sep)[-1]
        foldername=path_mr.split(os.sep)[-2]
        filename=filename.replace('MR','CT')
        foldername=foldername.replace('MR','CT')

        path_ct=os.path.join(self.dir_data,foldername,filename)
        # Loading Data and get label
        # print('Debug here')
        X = Image.open(path_mr)
        # print(np.array(X).shape)
        X = np.array(X)
        X = X[:160,:192] #values taken as a multiple of 32
        X = np.concatenate((X[:,:,np.newaxis],X[:,:,np.newaxis],X[:,:,np.newaxis]), axis=2)
        X = ToTensor()(X)
        y = Image.open(path_ct)
        y = np.array(y)[:160,:192,np.newaxis]
        y = ToTensor()(y)
        return X, y

#####

# class MRCT_Dataset(data.Dataset):
#     'characterizes a dataset for pytorch'
#     def __init__(self, dir_data, list_IDs):
#         self.list_IDs = list_IDs
#         self.dir_data=dir_data
#     def __len__(self):
#         'denotes the total number of samples'
#         return len(self.list_IDs)
#
#     def __getitem__(self, item):
#         'Generates one sameple of data'
#         #select sample
#         path_mr = os.path.join(self.dir_data,'mr',self.list_IDs[item])
#         path_ct = os.path.join(self.dir_data,'ct',self.list_IDs[item])
#         # Loading Data and get label
#         # print('Debug here')
#         X = Image.open(path_mr)
#         # print(np.array(X).shape)
#         X=np.array(X)
#         X=X[:160,:192]
#         X=np.concatenate((X[:,:,np.newaxis],X[:,:,np.newaxis],X[:,:,np.newaxis]), axis=2)
#         X = ToTensor()(X)
#         y = Image.open(path_ct)
#         y=np.array(y)[:160,:192]
#         y = ToTensor()(y)
#         return X, y

