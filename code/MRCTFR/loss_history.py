import os
import pickle
import matplotlib.pyplot as plt

dir_project = r'C:\Users\ranab\PycharmProjects\MRCTFR'
TIME_STAMP='2019-04-30-17-26-37'
path_log=os.path.join(dir_project,'log','{}.bin'.format(TIME_STAMP))
with open(path_log,'rb') as pfile:
    h=pickle.load(pfile)

print(h.keys())

plt.plot(h['loss_train'],'b',label='train')
plt.plot(h['loss_valid'],'r',label='valid')

plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()