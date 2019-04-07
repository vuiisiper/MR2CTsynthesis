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
    FILEPATH_LOG = os.path.join(dir_log, '{}.bin'.format(TIME_STAMP)) # save loss history here

    ## Training parameters

    max_epochs = 2
    lr = 0.001

    ## Setup loaders
    train_dataset = Dataset(dir_train)
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    print('train directory has {} samples'.format(len(train_loader)))

    valid_dataset = Dataset(dir_valid)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
    print('validation directory has {} samples'.format(len(valid_loader)))

    test_dataset = Dataset(dir_test)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print('test directory has {} samples'.format(len(test_loader)))

    model_save_criteria=np.inf # initialize the threshold for saving the model
    train_states={} # initialze a dict to save training state of the model

    model11 = unet11(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model11.parameters(), lr=lr)
    print('Debug here')
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
            mr = sample[0].to(device)
            ct = sample[1].float().to(device)
            ct_predict = model11(mr)
            optimizer.zero_grad()
            loss = criterion(ct, ct_predict)
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
                mr = sample[0].to(device)
                ct = sample[1].float().to(device)
                ct_predict = model11(mr)
                loss = criterion(ct, ct_predict)
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
            mr = sample[0].to(device)
            ct = sample[1].float().to(device)
            ct_predict = model11(mr)
            loss = criterion(ct, ct_predict)
            running_loss += loss.item()
            mean_loss = running_loss / (i + 1)
    print('test_loss {:.4f}'.format(mean_loss))
            # break


    # print('Debug here')
            # break

    # import matplotlib.pyplot as plt
    # plt.imshow(output)
    # plt.show()


