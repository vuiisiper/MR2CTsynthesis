import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

dir_mr = r'D:\Data\MR2CTsynthesis\MRCT_data_2D_256_256_tif\Test\MR'
dir_ct = r'D:\Data\MR2CTsynthesis\MRCT_data_2D_256_256_tif\Test\CT'
dir_pred = r'D:\Data\MR2CTsynthesis\output\2019-05-10-23-37-05'
filepaths_mr = glob.glob(os.path.join(dir_mr,'*'))
filepaths_ct = glob.glob(os.path.join(dir_ct,'*'))
# filepaths_pred = glob.glob(os.path.join(dir_pred,'*'))

## Setup loaders
for i, (path_mr, path_ct) in enumerate(zip(filepaths_mr,filepaths_ct)):
    mr = Image.open(path_mr)
    mr = np.array(mr)
    ct = Image.open(path_ct)
    ct = np.array(ct)
    pred = np.load(os.path.join(dir_pred,'{}.npy'.format(i)))
    plt.subplot(131)
    plt.imshow(mr, cmap = 'gray')
    plt.title('MR')
    plt.subplot(132)
    plt.imshow(ct,cmap = 'gray')
    plt.title('CT')
    plt.subplot(133)
    plt.imshow(pred,cmap = 'gray')
    plt.title('CT Prediction')
    plt.show()
