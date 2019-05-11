import matplotlib.pyplot as plt
import torch
from Networks import unet11
import os
from DataLoader import DatasetKspace
from torch.utils import data
import torch.nn as nn
from utils import mse_loss
from utils import mae_loss
from utils import psnr
from tqdm import tqdm
import argparse
import numpy as np
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument('--dirData', default=r'D:\PaddedMRCT')
parser.add_argument('--batchSize', default=, type = int)

args = parser.parse_args()

TIME_STAMP = '2019-04-30-17-30-58'
dir_data = args.dirData #D:\PaddedMRCT
with open(os.path.join(dir_data, 'Images', 'test.txt'), 'r') as f:
    filenames_test = f.readlines()
filenames_test = [item.strip() for item in filenames_test]
dir_test = os.path.join(dir_data,'Images', 'Test')
test_dataset = DatasetKspace(dir_test,filenames_test) #dir_test
test_loader = data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False)
print('test directory has {} samples'.format(len(test_dataset)))
dir_out = os.path.join(dir_data,'output',TIME_STAMP)
dir_metrics = os.path.join(dir_data,'metrics')  #todo: do we need it now? or after generating CT image domain?

# mr_test,ct_test = test_loader.dataset[10]
# mr_test = np.array(mr_test)
# mr_test = mr_test[np.newaxis,:,:,:]
# mr_test= torch.tensor(mr_test)

criterion = nn.MSELoss()
model11 = unet11().cuda()
FILEPATH_MODEL_LOAD = os.path.join(dir_data,'model','2019-04-30-01-58-58.pt') #todo: load the latest trained model
train_states = torch.load(FILEPATH_MODEL_LOAD)

model11.load_state_dict(train_states['train_states_best']['model_state_dict'])

MAE_list = []
MSE_list = []
PSNR_list = []

for i, sample in enumerate(tqdm(test_loader)):
    with torch.no_grad():
        mr_test = sample[0] #torch.Size([56, 3, 192, 224])
        # print(mr_test.min())
        ct_test = sample[1] #torch.Size([56, 1, 192, 224])
        pred = model11(mr_test.float().cuda()) #torch.Size([56, 1, 192, 224])
        pred_abs = pred.detach().cpu().numpy().transpose([0, 2, 3, 1]).squeeze() #torch.Size([56, 1, 192, 224])
        CT_test_squeezed = ct_test.numpy().transpose([0,2,3,1]).squeeze() #numpy.ndarray  (56, 192, 224)
        MR_np = mr_test.detach().cpu().numpy().transpose([0,2,3,1]).squeeze() # (56, 192, 224, 3)
        path_save = os.path.join(dir_out, 'mat_{}.mat'.format(os.path.splitext(filenames_test[i])[0]))
        savemat(path_save, {'mat_'+os.path.splitext(filenames_test[i])[0]: pred_abs})
        # for j,(input,output,target) in enumerate(zip(MR_np,CT_prediction,CT_test_squeezed)):
            # plt.tight_layout()
            # plt.subplot(131).imshow(input, cmap='gray')
            # plt.title('Input')
            # plt.axis('off')
            # plt.subplot(132).imshow(output,cmap='gray')
            # plt.title('Output \nMAE {:.2f}'.format(mae_loss(output,target)))
            # plt.axis('off')
            # plt.subplot(133).imshow(target,cmap='gray')
            # # print(mse_loss(output,target))
            # # print(mae_loss(output,target))
            # plt.title('Target')
            # # plt.show()
            # # plt.tight_layout()
            # plt.axis('off')
            # plt.savefig(os.path.join(dir_out,'{}.png'.format(j+64*i)), dpi=300,bbox_inches='tight')
            # MAE_list.append(mae_loss(output,target)) #from utils.py
            # MSE_list.append(mse_loss(output,target))
            # PSNR_list.append(psnr(output,target))
metrics_dict={
    'MAE_list' : MAE_list,
    'MSE_list' : MSE_list,
    'PSNR_list' : PSNR_list
}
# with open(os.path.join(dir_metrics,'{}_rotated.bin'.format(TIME_STAMP)),'wb') as pfile: #todo: commented out these two?
#     pickle.dump(metrics_dict, pfile)