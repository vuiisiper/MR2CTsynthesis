from DataLoader import DatasetKspace
from Networks import unet11
import torch
import torch.nn as nn
from torch.utils import data
from utils import Logger
import os
import argparse
import sys
import pickle
import time
import numpy as np

if __name__ == '__main__':
    """
    Run directly
    """
    ## timestamping the model and log files is a good practice for keeoing track of your experiments
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')  # year-month-day=hour-minute-second
    # dir_project='C:\\Users\\Reasat\\Projects\\MR2CTsynthesis'
    # dir_data='D:\\Data\\MR2CTsynthesis'
    dir_project = r'C:\Users\ranab\PycharmProjects\MRCTFR'
    dir_data = r'D:\PaddedMRCT' #where images are split into train, valid and test folders

    path_out = os.path.join(dir_project,'out','{}.out'.format(TIME_STAMP))

    sys.stdout = Logger(path_out)

    print(TIME_STAMP)
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch', type=int, help='epochs')
    parser.add_argument('batchSize', type=int, help='batch size')
    parser.add_argument('lr', type=float, help='learning rate')
    args = parser.parse_args()

    config = vars(args) #todo: why
    config_ls = sorted(list(config.items()))
    print(
        '--------------------------------------------------------------------------------------------------------------------')
    for item in config_ls: #todo: why printing these
        print('{}: {}'.format(item[0], item[1]))
    print(
        '--------------------------------------------------------------------------------------------------------------------')

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


    dir_model = os.path.join(dir_data, 'model')
    dir_log = os.path.join(dir_project, 'log')
    dir_train = os.path.join(dir_data, 'Images','Train')
    dir_test = os.path.join(dir_data, 'Images','Test')
    dir_valid = os.path.join(dir_data, 'Images','Valid')
    FILEPATH_MODEL_SAVE = os.path.join(dir_model, '{}.pt'.format(TIME_STAMP)) # save model here
    # FILEPATH_MODEL_LOAD = os.path.join(dir_model, '{}.pt'.format(TIME_STAMP)) # load model weights to resume training
    FILEPATH_MODEL_LOAD=None
    FILEPATH_LOG = os.path.join(dir_log, '{}.bin'.format(TIME_STAMP)) # save loss history here

    ## Training hyper-parameters

    max_epochs = args.epoch
    lr = args.lr

    ## Setup loaders
    with open(os.path.join(dir_data, 'Images','train.txt'), 'r') as f:
        filenames_train = f.readlines()
    filenames_train = [item.strip() for item in filenames_train]
    with open(os.path.join(dir_data, 'Images','valid.txt'), 'r') as f:
        filenames_valid = f.readlines()
    filenames_valid = [item.strip() for item in filenames_valid]
    with open(os.path.join(dir_data, 'Images','test.txt'), 'r') as f:
        filenames_test = f.readlines()
    filenames_test = [item.strip() for item in filenames_test]

    train_dataset = DatasetKspace(dir_train,filenames_train) #dir_train
    train_loader = data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True) #data comes from torch.utils
    print('train directory has {} samples'.format(len(train_dataset)))

    valid_dataset = DatasetKspace(dir_valid,filenames_valid) #dir_valid
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batchSize, shuffle=False)
    print('validation directory has {} samples'.format(len(valid_dataset)))

    test_dataset = DatasetKspace(dir_test,filenames_test) #dir_test
    test_loader = data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False)
    print('test directory has {} samples'.format(len(test_dataset)))

    model_save_criteria = np.inf # initialize the threshold for saving the model        #todo: এর মানে কি?
    train_states = {} # initialze a dict to save training state of the model

    model11 = unet11(pretrained=True).to(device) #unet11 imported from UNet_models.py
    criterion = nn.MSELoss() #loss will be calculated in Mean Square Error algorithm  per batch size?
    optimizer = torch.optim.Adam(model11.parameters(), lr=lr) #Optimization will be performed using Adam optimizer

    ##  resuming a training session
    if FILEPATH_MODEL_LOAD is not None:
        train_states = torch.load(FILEPATH_MODEL_LOAD)
        model11.load_state_dict(train_states['train_states_latest']['model_state_dict'])
        optimizer.load_state_dict(train_states['train_states_latest']['optimizer_state_dict'])
        train_states_best = train_states['train_states_best']
        # loss_valid_min=train_states_best['loss_valid_min'] # change
        model_save_criteria = train_states_best['model_save_criteria']  # change

    else:
        train_states={}
        model_save_criteria = np.inf            #todo: এর মানে কি?

    ## Train
    loss_epoch_train=[]
    loss_epoch_valid=[]

    for epoch in range(max_epochs):
        running_loss = 0
        running_time_batch = 0
        time_batch_start = time.time()
        model11.train()
        for i, sample in enumerate(train_loader):
            time_batch_load = time.time() - time_batch_start
            time_compute_start = time.time()
            mr = sample[0].float().to(device)
            abs = sample[1].float().to(device)   #todo: float or int16
            # phase = sample[2].float().to(device)   #todo: float or int16

            output = model11(mr)
            optimizer.zero_grad()
            loss = criterion(abs, output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            mean_loss = running_loss / (i + 1)

            # print time stats
            time_compute = time.time() - time_compute_start
            time_batch = time_batch_load + time_compute
            running_time_batch += time_batch
            time_batch_avg = running_time_batch / (i + 1)

            print(
                'epoch: {}/{}, batch: {}/{}, loss-train: {:.4f}, batch time taken: {:.2f}s, eta_epoch: {:.2f} hours'.format(
                    epoch + 1,
                    max_epochs,
                    i + 1,
                    len(train_loader),
                    mean_loss,
                    time_batch,
                    time_batch_avg * (len(train_loader) - (i + 1)) / 3600,
                )
            )
            time_batch_start=time.time()
        loss_epoch_train.append(mean_loss)

        ## Validation
        for i, sample in enumerate(valid_loader):
            running_loss = 0
            model11.eval()
            with torch.no_grad():
                mr = sample[0].float().to(device)
                abs = sample[1].float().to(device)
                output = model11(mr)
                loss = criterion(abs, output)
                running_loss += loss.item()
                mean_loss = running_loss / (i + 1)
                print(
                    'epoch: {}/{}, batch: {}/{}, loss-valid: {:.4f}'.format(
                        epoch + 1,
                        max_epochs,
                        i + 1,
                        len(valid_loader),
                        mean_loss,
                    )
                )
        loss_epoch_valid.append(mean_loss)

        ## Save model if loss decreases
        chosen_criteria = mean_loss
        print('criteria at the end of epoch {} is {:.4f}'.format(epoch + 1, chosen_criteria))

        if chosen_criteria < model_save_criteria:  # save model if true
            print('criteria decreased from {:.4f} to {:.4f}, saving model...'.format(model_save_criteria,
                                                                                     chosen_criteria))
            train_states_best = {
                'epoch': epoch + 1,
                'model_state_dict': model11.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_save_criteria': chosen_criteria,
            }
            train_states = {
                'train_states_best': train_states_best,
            }
            torch.save(train_states, FILEPATH_MODEL_SAVE)

            model_save_criteria = chosen_criteria

        log = {
            'loss_train': loss_epoch_train,
            'loss_valid': loss_epoch_valid,
        }
        with open(FILEPATH_LOG, 'wb') as pfile:
            pickle.dump(log, pfile)

        ## also save the latest model after each epoch as you may want to resume training at a later time
        train_states_latest = {
            'epoch': epoch + 1,
            'model_state_dict': model11.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_save_criteria': chosen_criteria,
        }
        train_states['train_states_latest'] = train_states_latest
        torch.save(train_states, FILEPATH_MODEL_SAVE)

    ## Test
    for i, sample in enumerate(test_loader):
        running_loss = 0
        model11.eval()  # sets the model in evaluation mode
        with torch.no_grad():
            mr = sample[0].float().to(device)
            abs = sample[1].float().to(device)
            output = model11(mr)
            loss = criterion(abs, output)
            running_loss += loss.item()
            mean_loss = running_loss / (i + 1)
    print('test_loss {:.4f}'.format(mean_loss))
            # break


    # print('Debug here')
            # break

    # import matplotlib.pyplot as plt
    # plt.imshow(output)
    # plt.show()


