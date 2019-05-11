import torch
from torch.utils import data
from DataLoader_tif import Dataset
from torch.backends import cudnn
import torch.nn as nn
import os
# from UNet_models import unet11
import time
import numpy as np
import pickle
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from Unet2d_pytorch_modified import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from nnBuildUnits import CrossEntropy3d, topK_RegLoss, RelativeThreshold_RegLoss, gdl_loss, adjust_learning_rate, calc_gradient_penalty
from utils import weights_init,Logger   #todo: not importing?
import argparse
if __name__ == '__main__':
    ## timestamping the model and log files is a good practice for keeping track of your experiments
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S') # year-month-day=hour-minute-second
    """
    Run directly
    """
    use_cuda = torch.cuda.is_available() #gives True if CUDA is available ->True
    device = torch.device("cuda:0"if use_cuda else "cpu") # -> cuda:0
    # cudnn.benchmark = True

    ## Setup directories

    # directory structure
    # ** Manually make these directories **
    # MR2CTsynthesis (Project directory for small files which will be pushed to git )
    #   --code (put scripts in the code folder)
    #   --log
    # MR2CTsynthesis (directory for large files which will be stored locally, too large for git)
    #   --MR_CT_data
    #     --Train
    #     --Valid
    #     --Test
    #   --model

    # dir_project=r'C:\Users\reasat\Projects\MR2CTsynthesis'
    # dir_lf = r'D:\Data\MR2CTsynthesis'
    # dir_data = os.path.join(dir_lf,'MRCT_data_2D_256_256_tif')
    parser = argparse.ArgumentParser()
    # parser.add_argument('dir_project', default = r'C:\Users\reasat\Projects\MR2CTsynthesis')
    parser.add_argument('--time_stamp', required = True)
    parser.add_argument('--epoch', type=int, required = True)
    parser.add_argument('--dir_lf', help = 'directory of large file, (data, trained, model, output)', default = r'D:\Data\MR2CTsynthesis')
    parser.add_argument('--folder_data', help = 'data folder e.g. MRCT_data_2D_256_256_tif', default = 'MRCT_data_2D_256_256_tif')
    parser.add_argument('--rt_th',type = int,default=0.005)
    parser.add_argument('--gdlNorm',type = int,default=2)
    parser.add_argument('--batch_size',type = int, required = True)
    parser.add_argument('--lambda_rec',type = float, required = True)
    parser.add_argument('--lambda_AD',type = float, required = True)
    parser.add_argument('--lambda_gdl',type = float, required = True)
    parser.add_argument('--numOfChannel_allSource',type = int,default=1)
    args = parser.parse_args()
    dir_data = os.path.join(args.dir_lf,args.folder_data)
    dir_model = os.path.join(args.dir_lf, 'model')
    dir_output = os.path.join(args.dir_lf, 'output')
    # dir_train = os.path.join(args.dir_data, 'Train')
    dir_test = os.path.join(dir_data, 'Test')
    # dir_valid = os.path.join(args.dir_data, 'Valid')
    filepath_model_load = os.path.join(dir_model, '{}_epoch_{}.pt'.format(args.time_stamp,args.epoch)) # load model weights to resume training

    print(args.time_stamp)
    config = vars(args)
    config_ls = sorted(list(config.items()))
    print(
        '--------------------------------------------------------------------------------------------------------------------')
    for item in config_ls:
        print('{}: {}'.format(item[0], item[1]))
    print(
        '--------------------------------------------------------------------------------------------------------------------')

    dir_gen = os.path.join(dir_output,args.time_stamp)
    if not os.path.exists(dir_gen):
        print('making directory to save model output in {}'.format(dir_gen))
        os.mkdir(dir_gen)

    gdlNorm = args.gdlNorm  #gradient difference loss normalized
    batch_size = args.batch_size
    lambda_rec = args.lambda_rec   #lambda2
    lambda_AD = args.lambda_AD  #lambda1
    lambda_gdl = args.lambda_gdl  #lambda3 in paper it was 1
    numOfChannel_allSource = args.numOfChannel_allSource

    ## Setup loaders
    test_dataset = Dataset(dir_test)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('test directory has {} samples'.format(len(test_dataset)))

    model_save_criteria = np.inf  #initialize the threshold for saving the model
    train_states = {}  #initialize a dict to save training state of the model

    netD = Discriminator()   # disc object
    netD.apply(weights_init) #where is weights_init declared? #todo ?
    netD.cuda()

    # optimizerD = optim.Adam(netD.parameters(), lr=lr_netD)
    criterion_bce = nn.BCELoss()   #binary cross entropy loss
    criterion_bce.cuda()

    netG = UNet(in_channel=numOfChannel_allSource, n_classes=1).to(device) # generator object
    params = list(netG.parameters())
    # print('len of params is ')
    # print(len(params))
    # print('size of params is ')
    # print(params[0].size())

    # optimizerG = optim.Adam(netG.parameters(), lr = lr_netG)
    criterion_L2 = nn.MSELoss()  #mse loss is called squared L2 norm
    criterion_L1 = nn.L1Loss()  #mean absolute error between target and output
    criterion_RTL1 = RelativeThreshold_RegLoss(args.rt_th)
    criterion_gdl = gdl_loss(gdlNorm)

    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_RTL1 = criterion_RTL1.cuda()
    criterion_gdl = criterion_gdl.cuda()
    ##  resuming a training session
    print('loading model from {}'.format(filepath_model_load))
    train_states = torch.load(filepath_model_load)
    netD.load_state_dict(train_states['model_d_state_dict'])
    # optimizerD.load_state_dict(train_states['optimizer_d_state_dict'])
    netG.load_state_dict(train_states['model_g_state_dict'])
    # optimizerG.load_state_dict(train_states['optimizer_g_state_dict'])

    loss_d_real_test = []
    loss_d_fake_test = []
    loss_g_test = []
    loss_g_g_test = []
    loss_g_d_test = []
    loss_gdl_test = []
    ## Evaluating
    print('evaluating ...')
    running_loss_d_real = 0
    running_loss_d_fake = 0
    running_loss_g = 0
    running_loss_g_g = 0  # gen reconstruction loss
    running_loss_g_d = 0  # discriminator advarsarial loss
    running_loss_gdl = 0  # gradient difference loss
    ind_sample=0
    for i, sample in enumerate(test_loader):
        netG.eval()
        netD.eval()
        with torch.no_grad():
            mr = sample[0].float().to(device)
            ct = sample[1].float().to(device)
            outputG = netG(mr)
            outputD = netD(outputG)
            outputD = F.sigmoid(outputD)
            batch_size_temp = len(sample[0])
            real_label = torch.ones(batch_size_temp, 1)  # create a vector of 1
            real_label = real_label.cuda()
            real_label = Variable(real_label)  # just changing the data structure or so
            fake_label = torch.zeros(batch_size_temp, 1)
            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            outputD_real = netD(ct)  # outputD_real =  predicted_label
            outputD_real = F.sigmoid(outputD_real)  # outputD_real is just probability [0,1]
            outputD_fake = netD(outputG)  # outputD_fake is a
            outputD_fake = F.sigmoid(outputD_fake)  #

            loss_real = criterion_bce(outputD_real, real_label)
            loss_fake = criterion_bce(outputD_fake, fake_label)

            lossD = loss_real + loss_fake
            lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(ct))
            lossG_G = lambda_rec * lossG_G
            lossG_D = lambda_AD * criterion_bce(outputD,real_label)  # note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_gdl = lambda_gdl * criterion_gdl(outputG, torch.unsqueeze(torch.squeeze(ct, 1), 1))
            lossG = lossG_G + lossG_D + lossG_gdl

            running_loss_d_real += loss_real.item()
            running_loss_d_fake += loss_fake.item()
            running_loss_g += lossG.item()
            running_loss_g_g += lossG_G.item()
            running_loss_g_d += lossG_D.item()
            running_loss_gdl += lossG_gdl.item()

            mean_loss_d_real = running_loss_d_real / (i + 1)
            mean_loss_d_fake = running_loss_d_fake / (i + 1)
            mean_loss_g = running_loss_g / (i + 1)
            mean_loss_g_g = running_loss_g_g / (i + 1)
            mean_loss_g_d = running_loss_g_d / (i + 1)
            mean_loss_gdl = running_loss_gdl / (i + 1)
            print('batch: {}/{}, lossD_real_avg: {:.4f}, lossD_fake_avg: {:.4f}, lossG_avg: {:.4f}, lossG_G_avg: {:.4f}, lossG_D_avg: {:.4f}, lossG_gdl_avg: {:.4f}'.format(
                    i + 1,
                    len(test_loader),
                    mean_loss_d_real,
                    mean_loss_d_fake,
                    mean_loss_g,
                    mean_loss_g_g,
                    mean_loss_g_d,
                    mean_loss_gdl,
                ),
                end='\r'
            )
        for item in outputG:
            np.save(os.path.join(dir_gen,'{}.npy'.format(ind_sample)), item.cpu().numpy().squeeze())
            ind_sample += 1

        loss_d_real_test.append(mean_loss_d_real)
        loss_d_fake_test.append(mean_loss_d_fake)
        loss_g_test.append(mean_loss_g)
        loss_g_g_test.append(mean_loss_g_g)
        loss_g_d_test.append(mean_loss_g_d)
        loss_gdl_test.append(mean_loss_gdl)

    losses = {
        'loss_d_test':loss_d_real_test,
        'loss_g_test':loss_g_test,
        'loss_g_g_test':loss_g_g_test,
        'loss_g_d_test':loss_g_d_test,
        'loss_gdl_test':loss_gdl_test,
    }
    with open(os.path.join(dir_gen,'metrics.bin'), 'wb') as pfile:
        pickle.dump(losses, pfile)


