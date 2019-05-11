import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from skimage.measure import compare_ssim as ssim

TIME_STAMP='2019-05-11-03-35-51'
dir_mr = r'D:\Data\MR2CTsynthesis\MRCT_data_2D_256_256_tif\Test\MR'
dir_ct = r'D:\Data\MR2CTsynthesis\MRCT_data_2D_256_256_tif\Test\CT'
dir_pred = os.path.join(r'D:\Data\MR2CTsynthesis\output',TIME_STAMP)
filepaths_mr = glob.glob(os.path.join(dir_mr,'*'))
filepaths_ct = glob.glob(os.path.join(dir_ct,'*'))
# filepaths_pred = glob.glob(os.path.join(dir_pred,'*'))
dir_plot = os.path.join(r'D:\Data\MR2CTsynthesis\visual_output',TIME_STAMP)
if not os.path.exists(dir_plot):
    print('making directory to save model output in {}'.format(dir_plot))
    os.mkdir(dir_plot)
## Setup loaders
ssim_list = []
for i, (path_mr, path_ct) in enumerate(zip(filepaths_mr,filepaths_ct)):
    mr = Image.open(path_mr)
    mr = np.array(mr)
    ct = Image.open(path_ct)
    ct = np.array(ct)
    pred = np.load(os.path.join(dir_pred,'{}.npy'.format(i)))
    ssim_sample = ssim(ct,pred)
    ssim_list.append(ssim_sample)
    patient_id = ((i+1)//(312/2))+1
    plt.subplot(131)
    plt.imshow(mr, cmap = 'gray')
    plt.title('MR')
    plt.subplot(132)
    plt.imshow(ct,cmap = 'gray')
    plt.title('CT')
    plt.subplot(133)
    plt.imshow(pred,cmap = 'gray')
    plt.title('CT Prediction')
    plt.suptitle('Test Patient {}, Slice {}, SSIM {:.4f}'.format(patient_id,i,ssim_sample))
    plt.savefig()
    # plt.show()
