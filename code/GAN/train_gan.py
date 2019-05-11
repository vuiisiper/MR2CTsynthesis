import torch
from torch.utils import data
from DataLoader_tif import Dataset
from torch.backends import cudnn
import torch.nn as nn
import os
import sys
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
    ## timestamping the model and log files is a good practice for keeoing track of your experiments
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S') # year-month-day=hour-minute-second
    print(TIME_STAMP)
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
    parser.add_argument('--dir_project', default = r'C:\Users\reasat\Projects\MR2CTsynthesis')
    parser.add_argument('--dir_lf', help = 'directory of large file, (data, trained, model, output)', default = r'D:\Data\MR2CTsynthesis')
    parser.add_argument('--folder_data', help = 'data folder e.g. MRCT_data_2D_256_256_tif', default = 'MRCT_data_2D_256_256_tif')
    parser.add_argument('--max_epochs',type = int, required= True)
    parser.add_argument('--save_epoch',type = int, required= True)
    parser.add_argument('--lr_netG',type = float, required= True)
    parser.add_argument('--lr_netD',type = float, required= True)
    parser.add_argument('--rt_th',type = int,default=0.005)
    parser.add_argument('--gdlNorm',type = int,default=2)
    parser.add_argument('--batch_size',type = int, required= True)
    parser.add_argument('--lambda_rec',type = float, required= True)
    parser.add_argument('--lambda_AD',type = float, required= True)
    parser.add_argument('--lambda_gdl',type = float, required= True)
    parser.add_argument('--numOfChannel_allSource',type = int,default=1)
    parser.add_argument('--pretrained')
    args = parser.parse_args()
    dir_data = os.path.join(args.dir_lf,args.folder_data)
    dir_model = os.path.join(args.dir_lf, 'model')
    dir_log = os.path.join(args.dir_project, 'log')
    dir_train = os.path.join(dir_data, 'Train')
    dir_test = os.path.join(dir_data, 'Test')
    dir_valid = os.path.join(dir_data, 'Valid')
    filepath_log = os.path.join(dir_log, '{}.bin'.format(TIME_STAMP)) # save loss history here
    filepath_out = os.path.join(args.dir_project,'out','{}.out'.format(TIME_STAMP))
    filepath_cfg = os.path.join(args.dir_project,'config','{}.cfg'.format(TIME_STAMP))

    if args.pretrained is not None:
        filepath_model_load = os.path.join(dir_model, '{}.pt'.format(TIME_STAMP)) # load model weights to resume training
    else:
        filepath_model_load=None

    sys.stdout = Logger(filepath_out)
    config = vars(args)

    config_ls = sorted(list(config.items()))
    print(
        '--------------------------------------------------------------------------------------------------------------------')
    for item in config_ls:
        print('{}: {}'.format(item[0], item[1]))
    print(
        '--------------------------------------------------------------------------------------------------------------------')
    with open(filepath_cfg,'w') as file:
        for item in config_ls:
            file.write('{}: {}\n'.format(item[0], item[1]))
    ## Training parameters

    max_epochs = args.max_epochs
    save_epoch = args.save_epoch
    lr_netG = args.lr_netG
    lr_netD = args.lr_netD
    RT_th = args.rt_th  #relative threshold
    gdlNorm = args.gdlNorm  #gradient difference loss normalized
    batch_size = args.batch_size
    lambda_rec = args.lambda_rec   #lambda2
    lambda_AD = args.lambda_AD  #lambda1
    lambda_gdl = args.lambda_gdl  #lambda3 in paper it was 1
    numOfChannel_allSource = args.numOfChannel_allSource

    ## Setup loaders
    train_dataset = Dataset(dir_train)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('train directory has {} samples'.format(len(train_dataset)))

    valid_dataset = Dataset(dir_valid)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print('validation directory has {} samples'.format(len(valid_dataset)))

    test_dataset = Dataset(dir_test)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('test directory has {} samples'.format(len(test_dataset)))

    model_save_criteria = np.inf  #initialize the threshold for saving the model
    train_states = {}  #initialize a dict to save training state of the model

    netD = Discriminator()   # disc object
    netD.apply(weights_init) #where is weights_init declared? #todo ?
    netD.cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=lr_netD)
    criterion_bce = nn.BCELoss()   #binary cross entropy loss
    criterion_bce.cuda()

    netG = UNet(in_channel=numOfChannel_allSource, n_classes=1).to(device) # generator object
    params = list(netG.parameters())
    # print('len of params is ')
    # print(len(params))
    # print('size of params is ')
    # print(params[0].size())

    optimizerG = optim.Adam(netG.parameters(), lr = lr_netG)
    criterion_L2 = nn.MSELoss()  #mse loss is called squared L2 norm
    criterion_L1 = nn.L1Loss()  #mean absolute error between target and output
    criterion_RTL1 = RelativeThreshold_RegLoss(RT_th)
    criterion_gdl = gdl_loss(gdlNorm)

    given_weight = torch.cuda.FloatTensor([1, 4, 4, 2]) # probably not used...

    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_RTL1 = criterion_RTL1.cuda()
    criterion_gdl = criterion_gdl.cuda()
    ##  resuming a training session
    if filepath_model_load is not None:
        print('loading models from {}'.format(filepath_model_load))
        train_states = torch.load(filepath_model_load)
        netD.load_state_dict(train_states['model_d_state_dict'])
        optimizerD.load_state_dict(train_states['optimizer_d_state_dict'])
        netG.load_state_dict(train_states['model_g_state_dict'])
        optimizerG.load_state_dict(train_states['optimizer_g_state_dict'])

    ## Train
    loss_d_real_epoch_train=[]
    loss_d_fake_epoch_train=[]
    loss_g_epoch_train =[]
    loss_g_g_epoch_train=[]  #generator generator loss
    loss_g_d_epoch_train=[]
    loss_gdl_epoch_train=[]
    loss_d_real_epoch_valid = []
    loss_d_fake_epoch_valid = []
    loss_g_epoch_valid = []
    loss_g_g_epoch_valid =[]
    loss_g_d_epoch_valid =[]
    loss_gdl_epoch_valid= []
    for epoch in range(max_epochs):
        print('training ...')
        running_loss_d_real = 0
        running_loss_d_fake = 0
        running_loss_g = 0
        running_loss_g_g = 0  #gen reconstruction loss
        running_loss_g_d = 0  #discriminator advarsarial loss
        running_loss_gdl = 0  #gradient difference loss
        running_time_batch = 0
        time_batch_start = time.time()
        for i, sample in enumerate(train_loader):
            netG.train()
            netD.train()
            time_batch_load = time.time() - time_batch_start
            time_compute_start = time.time()
            mr = sample[0].float().to(device) #[5, 1, 192, 256]
            # print('input sample shape of train_loader: {}'.format(mr.shape))
            ct = sample[1].float().to(device)
            batch_size_temp = len(sample[0])
            ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
            outputG = netG(mr) #outputG is an image which is generated CT
            outputD_real = netD(ct) # discriminator takes two input target/real CT, synthetic/fake CT
            outputD_real = F.sigmoid(outputD_real) #outputD_real is just probability [0,1]
            outputD_fake = netD(outputG.detach())  #outputD_fake is a
            outputD_fake = F.sigmoid(outputD_fake) #
            netD.zero_grad() #Sets gradients of all model parameters to zero. kind of reset?
            real_label = torch.ones(batch_size_temp, 1) #create a vector of 1
            real_label = real_label.cuda()
            real_label = Variable(real_label) #just changing the data structure or so/autograd, no longer used in pytorch
            # print(outputD_real.size())
            loss_real = criterion_bce(outputD_real, real_label) #first part of eqn.2
            # loss_real.backward()
            # train with fake data
            fake_label = torch.zeros(batch_size_temp, 1)
            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            loss_fake = criterion_bce(outputD_fake, fake_label) #second part of eqn.2
            # loss_fake.backward()

            lossD = loss_real + loss_fake  #eqn. 2 for discriminator
            lossD.backward(retain_graph=True)
            #             print 'loss_real is ',loss_real.data[0],'loss_fake is ',loss_fake.data[0],'outputD_real is',outputD_real.data[0]
            #             print('loss for discriminator is %f'%lossD.data[0])
            # update network parameters
            optimizerD.step()  #after propagating the loss backward you have to optimize the network(D) parameters

            ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
            # outputG = netG(mr) #synthetic or fake
            netG.zero_grad() #Sets gradients of all model parameters to zero. / reset?
            lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(ct))  #MSEloss, eqn. 1
            lossG_G = lambda_rec * lossG_G  #lambda2 = lambda2??
            # lossG_G.backward(retain_graph=True)  # compute gradients

            # outputG = netG(mr)
            outputD_fake = netD(outputG)  # outputD_fake
            outputD_fake = F.sigmoid(outputD_fake)
            lossG_D = lambda_AD * criterion_bce(outputD_fake,real_label) #Lambda1 * L_ADV;    # note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_gdl = lambda_gdl * criterion_gdl(outputG, torch.unsqueeze(torch.squeeze(ct, 1), 1))
            # lossG_gdl.backward()  # compute gradients

            lossG = lossG_G + lossG_D + lossG_gdl
            lossG.backward()
            optimizerG.step()

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

            # print time stats
            time_compute = time.time() - time_compute_start
            time_batch = time_batch_load + time_compute
            running_time_batch += time_batch
            time_batch_avg = running_time_batch / (i + 1)

            print(
                'epoch: {}/{}, batch: {}/{}, lossD_real_avg: {:.4f}, lossD_fake_avg: {:.4f}, lossG_avg: {:.4f}, lossG_G_avg: {:.4f}, lossG_D_avg: {:.4f}, lossG_gdl_avg: {:.4f}, eta_epoch: {:.2f} hours,'.format(
                    epoch + 1,
                    max_epochs,
                    i + 1,
                    len(train_loader),
                    mean_loss_d_real,
                    mean_loss_d_fake,
                    mean_loss_g,
                    mean_loss_g_g,
                    mean_loss_g_d,
                    mean_loss_gdl,
                    time_batch_avg * (len(train_loader) - (i + 1)) / 3600,
                ),
            )
            time_batch_start=time.time()

        loss_d_real_epoch_train.append(mean_loss_d_real)
        loss_d_fake_epoch_train.append(mean_loss_d_fake)
        loss_g_epoch_train.append(mean_loss_g)
        loss_g_g_epoch_train.append(mean_loss_g_g)
        loss_g_d_epoch_train.append(mean_loss_g_d)
        loss_gdl_epoch_train.append(mean_loss_gdl)

        ## Validation
        print('validating ...')
        running_loss_d_real = 0
        running_loss_d_fake = 0
        running_loss_g = 0
        running_loss_g_g = 0  # gen reconstruction loss
        running_loss_g_d = 0  # discriminator advarsarial loss
        running_loss_gdl = 0  # gradient difference loss
        for i, sample in enumerate(valid_loader):
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

                print(
                    'epoch: {}/{}, batch: {}/{}, lossD_real_avg: {:.4f}, lossD_fake_avg: {:.4f}, lossG_avg: {:.4f}, lossG_G_avg: {:.4f}, lossG_D_avg: {:.4f}, lossG_gdl_avg: {:.4f}'.format(
                        epoch + 1,
                        max_epochs,
                        i + 1,
                        len(train_loader),
                        mean_loss_d_real,
                        mean_loss_d_fake,
                        mean_loss_g,
                        mean_loss_g_g,
                        mean_loss_g_d,
                        mean_loss_gdl,
                    ),
                    end='\r'
                )

        loss_d_real_epoch_valid.append(mean_loss_d_real)
        loss_d_fake_epoch_valid.append(mean_loss_d_fake)
        loss_g_epoch_valid.append(mean_loss_g)
        loss_g_g_epoch_valid.append(mean_loss_g_g)
        loss_g_d_epoch_valid.append(mean_loss_g_d)
        loss_gdl_epoch_valid.append(mean_loss_gdl)


        log = {
            'loss_d_real_epoch_train':loss_d_real_epoch_train,
            'loss_d_fake_epoch_train':loss_d_fake_epoch_train,
            'loss_g_epoch_train':loss_g_epoch_train,
            'loss_g_g_epoch_train':loss_g_g_epoch_train,
            'loss_g_d_epoch_train':loss_g_d_epoch_train,
            'loss_gdl_epoch_train':loss_gdl_epoch_train,
            'loss_d_epoch_valid':loss_d_real_epoch_valid,
            'loss_g_epoch_valid':loss_g_epoch_valid,
            'loss_g_g_epoch_valid':loss_g_g_epoch_valid,
            'loss_g_d_epoch_valid':loss_g_d_epoch_valid,
            'loss_gdl_epoch_valid':loss_gdl_epoch_valid,
        }
        with open(filepath_log, 'wb') as pfile:
            pickle.dump(log, pfile)

        ## Save model if loss decreases
        # chosen_criteria = mean_loss
        # print('criteria at the end of epoch {} is {:.4f}'.format(epoch + 1, chosen_criteria))

        # if chosen_criteria < model_save_criteria:  # save model if true
        if (epoch + 1) % save_epoch == 0:
            # print('criteria decreased from {:.4f} to {:.4f}, saving model...'.format(model_save_criteria,
            #                                                                          chosen_criteria))
            print('saving model at epoch {}'.format(epoch+1))
            train_states = {
                'epoch': epoch + 1,
                'model_d_state_dict': netD.state_dict(),
                'optimizer_d_state_dict': optimizerD.state_dict(),
                'model_g_state_dict': netG.state_dict(),
                'optimizer_g_state_dict': optimizerG.state_dict(),
            }

            torch.save(train_states,  os.path.join(dir_model, '{}_epoch_{}.pt'.format(TIME_STAMP,epoch+1))) # save model here


        # ## also save the latest model after each epoch as you may want to resume training at a later time
        # train_states_latest = {
        #     'epoch': epoch + 1,
        #     'model_state_dict': model11.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'model_save_criteria': chosen_criteria,
        # }
        # train_states['train_states_latest'] = train_states_latest
        # torch.save(train_states, FILEPATH_MODEL_SAVE)

    # ## Test
    # for i, sample in enumerate(test_loader):
    #     running_loss = 0
    #     model11.eval()  # sets the model in evaluation mode
    #     with torch.no_grad():
    #         mr = sample[0].to(device)
    #         ct = sample[1].float().to(device)
    #         ct_predict = model11(mr)
    #         loss = criterion(ct, ct_predict)
    #         running_loss += loss.item()
    #         mean_loss = running_loss / (i + 1)
    # print('test_loss {:.4f}'.format(mean_loss))
    #         # break


    # print('Debug here')
            # break

    # import matplotlib.pyplot as plt
    # plt.imshow(output)
    # plt.show()


