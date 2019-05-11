import os
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils import data
from PIL import Image
import glob

def read_data_paths_form_dict():

    ct_data_path = r'D:\PaddedMRCT\Images\Train\CT' #abs CT image
    mri_data_path = r'D:\PaddedMRCT\Images\Train\MR'

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


def train_data_paths_form_list():

    ct_data_path = r'D:\PaddedMRCT\Images\Train\CT'
    mri_data_path = r'D:\PaddedMRCT\Images\Train\MR'

    ct_data_path = os.path.normpath(ct_data_path)
    mri_data_path = os.path.normpath(mri_data_path)

    mri_list_dir = os.listdir(mri_data_path)
    ct_list_dir = os.listdir(ct_data_path)

    total_samples = len(mri_list_dir)
    x_train = []
    y_train = []

    for i in range(total_samples):

        # print(i)
        mr_f_path = os.path.join(mri_data_path, mri_list_dir[i])
        ct_f_path = os.path.join(ct_data_path, ct_list_dir[i])

        x_train.append(mr_f_path)
        y_train.append(ct_f_path)

    return x_train, y_train


def test_data_paths_form_list():

    ct_test_data_path = r'D:\PaddedMRCT\Images\Test\CT'
    mr_test_data_path = r'D:\PaddedMRCT\Images\Test\MR'

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

def valid_data_paths_from_list():

    ct_valid_data_path = r'D:\PaddedMRCT\Images\Valid\CT'
    mr_valid_data_path = r'D:\PaddedMRCT\Images\Valid\MR'

    ct_valid_data_path = os.path.normpath(ct_valid_data_path)
    mr_valid_data_path = os.path.normpath(mr_valid_data_path)

    ct_valid_listdir = os.listdir(ct_valid_data_path)
    mr_valid_listdir = os.listdir(mr_valid_data_path)

    valid_samples = len(mr_valid_listdir)

    x_valid = []
    y_valid = []

    for i in range(valid_samples):

        mr_valid_path = os.path.join(mr_valid_data_path,mr_valid_listdir[i])
        ct_valid_path = os.path.join(ct_valid_data_path,ct_valid_listdir[i])

        x_valid.append(mr_valid_path)
        y_valid.append(ct_valid_path)

    return x_valid, y_valid

class Dataset(data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, dir, list_IDs, labels): #dir_data = r'D:\PaddedMRCT\Images'
        self.dir = dir
        self.list_IDs = list_IDs
        self.labels = labels
        # print(dir_data)
        # print(os.path.join(self.dir_data,'MR*','*.tif'))
        # self.filespaths_mr = glob.glob(os.path.join(self.dir, 'MR', '*.tif'))

    def __len__(self):
        'denotes the total number of samples'
        # print(len(self.filespaths_mr))
        # return len(self.filespaths_mr)
        return len(self.list_IDs)

    def __getitem__(self, item):
        'Generates one sample of data'
        #select sample
        # print(item)
        # path_mr = self.filespaths_mr[item]
        path_mr = os.path.join(self.list_IDs[item])
        path_ct = os.path.join(self.labels[item])
        # print(path_mr)
        # print(path_ct)
        # path_mr = os.path.join(self.dir,'MR', 'MR_{0:04}.tif'.format(item))
        # filename = path_mr.split(os.sep)[-1]  #takes the name of the file from lastest argument e.g. -1
        # foldername = path_mr.split(os.sep)[-2] #takes folder name from second lastest e.g. -2
        # filename = filename.replace('MR', 'CT')
        # foldername = foldername.replace('MR', 'CT')
        # path_ct = os.path.join(self.dir_data,foldername,filename)
        # path_ct = os.path.join(self.dir, 'CT', 'CT_{0:04}_abs.tif'.format(item))

        # Loading Data and get label
        X = Image.open(path_mr)
        X = np.array(X)
        X = np.concatenate((X[:,:,np.newaxis], X[:, :, np.newaxis], X[:, :, np.newaxis]), axis=2)
        X = ToTensor()(X)
        y = Image.open(path_ct)
        y = ToTensor()(y)
        return X, y

class DatasetKspace(data.Dataset):
    'characterizes a dataset for pytorch'
    def __init__(self, dir, list_IDs): #dir_data = r'D:\PaddedMRCT\Images\Train'
        self.dir = dir
        self.list_IDs = list_IDs
        # print(dir_data)
        # print(os.path.join(self.dir_data,'MR*','*.tif'))
        # self.filespaths_mr = glob.glob(os.path.join(self.dir, 'MR', '*.tif'))

    def __len__(self):
        'denotes the total number of samples'
        # print(len(self.filespaths_mr))
        # return len(self.filespaths_mr)
        return len(self.list_IDs)

    def __getitem__(self, item):
        'Generates one sample of data'
        #select sample
        # print(item)
        # path_mr = self.filespaths_mr[item]
        path_mr = os.path.join(self.dir, 'MR', self.list_IDs[item])
        path_abs = os.path.join(self.dir, 'CT', 'abs', self.list_IDs[item])
        path_phase = os.path.join(self.dir, 'CT', 'phase', self.list_IDs[item])

        # print(path_mr)
        # print(path_ct)
        # path_mr = os.path.join(self.dir,'MR', 'MR_{0:04}.tif'.format(item))
        # filename = path_mr.split(os.sep)[-1]  #takes the name of the file from lastest argument e.g. -1
        # foldername = path_mr.split(os.sep)[-2] #takes folder name from second lastest e.g. -2
        # filename = filename.replace('MR', 'CT')
        # foldername = foldername.replace('MR', 'CT')
        # path_ct = os.path.join(self.dir_data,foldername,filename)
        # path_ct = os.path.join(self.dir, 'CT', 'CT_{0:04}_abs.tif'.format(item))

        # Loading Data and get label
        X = Image.open(path_mr)
        X = np.array(X)
        X = np.concatenate((X[:,:,np.newaxis], X[:, :, np.newaxis], X[:, :, np.newaxis]), axis=2)
        X = ToTensor()(X)
        y_abs = Image.open(path_abs)
        y_phase = Image.open(path_phase)

        y_abs = ToTensor()(y_abs)
        y_phase = ToTensor()(y_phase)

        return X, y_abs, y_phase

#####
#This are not required?
# class MRCT_Dataset(data.Dataset):
#     'characterizes a dataset for pytorch'
#     def __init__(self, dir_data, list_IDs):
        # self.list_IDs = list_IDs
        # self.dir_data=dir_data
    # def __len__(self):
#         'denotes the total number of samples'
#         return len(self.list_IDs)
#
#     def __getitem__(self, item):
#         'Generates one sample of data'
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

