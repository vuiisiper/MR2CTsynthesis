import torch
from torch.utils import data
from DataLoader_tif import Dataset
from DataLoader_tif import read_data_paths_form_list, test_data_paths_form_list
from torch.backends import cudnn
import torch.nn as nn

if __name__ == '__main__':
    """
    Run directly
    """
    use_cuda = torch.cuda.is_available() #gives True if CUDA is available ->True
    device = torch.device("cuda:0"if use_cuda else "cpu") # -> cuda:0
    # cudnn.benchmark = True

    params = {'batch_size': 39,   #39 cause 2808 is divisible by 39 not necessarily
              'shuffle': True,
              'num_workers':1}
    max_epochs = 200
    lr = 0.001

    X_paths, y_paths = read_data_paths_form_list()
    train_dataset = Dataset(X_paths, y_paths)
    train_loader = data.DataLoader(train_dataset, **params)

    X_test, y_test = test_data_paths_form_list()
    test_dataset = Dataset(X_test, y_test)
    test_loader = data.DataLoader(test_dataset, **params)

    from UNet_models import unet11

    model11 = unet11(pretrained=True).to(device)
    model11.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model11.parameters(), lr=lr)
    print('Debug here')

    for j in range(max_epochs):
        for i, sample in enumerate(train_loader):
            mr = sample[0].to(device)
            ct = sample[1].float().to(device)
            ct_predict = model11(mr)
            optimizer.zero_grad()
            loss = criterion(ct, ct_predict)
            loss.backward()
            optimizer.step()
            print(i, loss.item()) #MSE loss is calculated per batch(over 39 images)
            if (i+1)%40 == 0:
                print('Epoch: {}/{} Loss: {}'.format(j+1,max_epochs,loss.item()))

        # print('Debug here')


    model11.eval() #sets the model in evaluation mode
    for i, sample in enumerate(test_loader):
        with torch.no_grad():
            mr = sample[0].to(device)
            ct = sample[1].float().to(device)
            ct_pred = model11(mr)
            output = ct_pred.detach().cpu().numpy().transpose([0,2,3,1])
            # break


    # print('Debug here')
            # break

    # import matplotlib.pyplot as plt
    # plt.imshow(output)
    # plt.show()

    torch.save(model11.state_dict(),'UMRCT.pt')

