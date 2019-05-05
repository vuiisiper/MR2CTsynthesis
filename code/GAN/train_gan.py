import torch
from torch.utils import data
from DataLoader_tif import Dataset
from torch.backends import cudnn
import torch.nn as nn
import os
from UNet_models import unet11
import time
import numpy as np
import pickle
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from medSynthesisV1.Unet2d_pytorch_modified import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from medSynthesisV1.nnBuildUnits import CrossEntropy3d, topK_RegLoss, RelativeThreshold_RegLoss, gdl_loss, adjust_learning_rate, calc_gradient_penalty
from medSynthesisV1.utils import weights_init

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
    # Data (directory for large files which will be stored locally, too large for git)
    #   --MR_CT_data
    #     --Train
    #     --Valid
    #     --Test
    #   --model

    # C:\Users\Reasat\Projects\MR2CTsynthesis\MR_CT_data\Valid\MR_test
    dir_project='C:\\Users\\Reasat\\Projects\\MR2CTsynthesis'
    dir_data='D:\\Data\\MR2CTsynthesis'
    dir_model = os.path.join(dir_data, 'model')
    dir_log = os.path.join(dir_project, 'log')
    dir_train = os.path.join(dir_data,'MR_CT_data', 'Train')
    dir_test = os.path.join(dir_data,'MR_CT_data', 'Test')
    dir_valid = os.path.join(dir_data,'MR_CT_data', 'Valid')
    FILEPATH_MODEL_SAVE = os.path.join(dir_model, '{}.pt'.format(TIME_STAMP)) # save model here
    # FILEPATH_MODEL_LOAD = os.path.join(dir_model, '{}.pt'.format(TIME_STAMP)) # load model weights to resume training
    FILEPATH_MODEL_LOAD=None
    FILEPATH_LOG = os.path.join(dir_log, '{}.bin'.format(TIME_STAMP)) # save loss history here

    ## Training parameters

    max_epochs = 2
    lr = 0.001
    lr_netD = 0.001
    RT_th = 0.005
    gdlNorm = 2
    batch_size = 5
    lossBase= 1
    lambda_AD = 0.05
    lambda_gdl = 0.05
    numOfChannel_allSource = 1

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

    model_save_criteria=np.inf # initialize the threshold for saving the model
    train_states={} # initialze a dict to save training state of the model

    netD = Discriminator()
    netD.apply(weights_init) #where is weights_init declared?
    netD.cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=lr_netD)
    criterion_bce = nn.BCELoss()
    criterion_bce.cuda()

    net = UNet(in_channel=numOfChannel_allSource, n_classes=1).to(device)
    params = list(net.parameters())
    # print('len of params is ')
    # print(len(params))
    # print('size of params is ')
    # print(params[0].size())

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion_L2 = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    criterion_RTL1 = RelativeThreshold_RegLoss(RT_th)
    criterion_gdl = gdl_loss(gdlNorm)

    given_weight = torch.cuda.FloatTensor([1, 4, 4, 2])

    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_RTL1 = criterion_RTL1.cuda()
    criterion_gdl = criterion_gdl.cuda()
    # ##  resuming a training session
    # if FILEPATH_MODEL_LOAD is not None:
    #     train_states = torch.load(FILEPATH_MODEL_LOAD)
    #     model11.load_state_dict(train_states['train_states_latest']['model_state_dict'])
    #     optimizer.load_state_dict(train_states['train_states_latest']['optimizer_state_dict'])
    #     train_states_best = train_states['train_states_best']
    #     # loss_valid_min=train_states_best['loss_valid_min'] # change
    #     model_save_criteria = train_states_best['model_save_criteria']  # change
    #
    # else:
    #     train_states={}
    #     model_save_criteria = np.inf

    ## Train
    print('training ...')
    loss_epoch_train=[]
    loss_epoch_valid=[]
    for epoch in range(max_epochs):
        running_loss_d = 0
        running_loss_g = 0
        running_loss_g_g =0
        running_loss_g_d = 0
        running_loss_gdl = 0
        running_time_batch = 0
        time_batch_start = time.time()
        net.train()
        for i, sample in enumerate(train_loader):
            time_batch_load = time.time() - time_batch_start
            time_compute_start = time.time()
            mr = sample[0].to(device)
            ct = sample[1].float().to(device)
            ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
            outputG = net(mr)
            outputD_real = netD(ct)
            outputD_real = F.sigmoid(outputD_real)
            outputD_fake = netD(outputG)
            outputD_fake = F.sigmoid(outputD_fake)
            netD.zero_grad()
            real_label = torch.ones(batch_size, 1)
            real_label = real_label.cuda()
            # print(real_label.size())
            real_label = Variable(real_label)
            # print(outputD_real.size())
            loss_real = criterion_bce(outputD_real, real_label)
            loss_real.backward()
            # train with fake data
            fake_label = torch.zeros(batch_size, 1)
            #         fake_label = torch.FloatTensor(batch_size)
            #         fake_label.data.resize_(batch_size).fill_(0)
            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            loss_fake = criterion_bce(outputD_fake, fake_label)
            loss_fake.backward()

            lossD = loss_real + loss_fake
            #             print 'loss_real is ',loss_real.data[0],'loss_fake is ',loss_fake.data[0],'outputD_real is',outputD_real.data[0]
            #             print('loss for discriminator is %f'%lossD.data[0])
            # update network parameters
            optimizerD.step()

            ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
            outputG = net(mr) # why get output again?
            net.zero_grad()
            lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(ct))
            lossG_G = lossBase * lossG_G
            # lossG_G.backward(retain_graph=True)  # compute gradients

            outputG = net(mr)
            outputD = netD(outputG)
            outputD = F.sigmoid(outputD)
            lossG_D = lambda_AD * criterion_bce(outputD,
                                                    real_label)  # note, for generator, the label for outputG is real, because the G wants to confuse D
            # lossG_D.backward()

            lossG_gdl = lambda_gdl * criterion_gdl(outputG, torch.unsqueeze(torch.squeeze(ct, 1), 1))
            # lossG_gdl.backward()  # compute gradients

            lossG = lossG_G +lossG_D + lossG_D + lossG_gdl
            lossG.backward()
            optimizer.step()

            running_loss_d += lossD.item()
            running_loss_g += lossG.item()
            running_loss_g_g += lossG_G.item()
            running_loss_g_d += lossG_D.item()
            running_loss_gdl += lossG_gdl.item()

            mean_loss_d = running_loss_d / (i + 1)
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
                'epoch: {}/{}, batch: {}/{}, lossD_avg: {:.4f}, lossG_avg: {:.4f}, lossG_G_avg: {:.4f}, lossG_D_avg: {:.4f}, lossG_gdl_avg: {:.4f}, eta_epoch: {:.2f} hours'.format(
                    epoch + 1,
                    max_epochs,
                    i + 1,
                    len(train_loader),
                    mean_loss_d,
                    mean_loss_g,
                    mean_loss_g_g,
                    mean_loss_g_d,
                    mean_loss_gdl,
                    # time_batch,
                    time_batch_avg * (len(train_loader) - (i + 1)) / 3600,
                )
            )
            time_batch_start=time.time()
        # loss_epoch_train.append(mean_loss)

    #     ## Validation
    #     for i, sample in enumerate(valid_loader):
    #         running_loss = 0
    #         model11.eval()
    #         with torch.no_grad():
    #             mr = sample[0].to(device)
    #             ct = sample[1].float().to(device)
    #             ct_predict = model11(mr)
    #             loss = criterion(ct, ct_predict)
    #             running_loss += loss.item()
    #             mean_loss = running_loss / (i + 1)
    #             print(
    #                 'epoch: {}/{}, batch: {}/{}, loss-valid: {:.4f}'.format(
    #                     epoch + 1,
    #                     max_epochs,
    #                     i + 1,
    #                     len(valid_loader),
    #                     mean_loss,
    #                 )
    #             )
    #     loss_epoch_valid.append(mean_loss)
    #
    #     ## Save model if loss decreases
    #     chosen_criteria = mean_loss
    #     print('criteria at the end of epoch {} is {:.4f}'.format(epoch + 1, chosen_criteria))
    #
    #     if chosen_criteria < model_save_criteria:  # save model if true
    #         print('criteria decreased from {:.4f} to {:.4f}, saving model...'.format(model_save_criteria,
    #                                                                                  chosen_criteria))
    #         train_states_best = {
    #             'epoch': epoch + 1,
    #             'model_state_dict': model11.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'model_save_criteria': chosen_criteria,
    #         }
    #         train_states = {
    #             'train_states_best': train_states_best,
    #         }
    #         torch.save(train_states, FILEPATH_MODEL_SAVE)
    #
    #         model_save_criteria = chosen_criteria
    #
    #     log = {
    #         'loss_train': loss_epoch_train,
    #         'loss_valid': loss_epoch_valid,
    #     }
    #     with open(FILEPATH_LOG, 'wb') as pfile:
    #         pickle.dump(log, pfile)
    #
    #     ## also save the latest model after each epoch as you may want to resume training at a later time
    #     train_states_latest = {
    #         'epoch': epoch + 1,
    #         'model_state_dict': model11.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'model_save_criteria': chosen_criteria,
    #     }
    #     train_states['train_states_latest'] = train_states_latest
    #     torch.save(train_states, FILEPATH_MODEL_SAVE)
    #
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


