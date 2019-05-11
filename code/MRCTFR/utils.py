import sys
import numpy as np
class Logger(object):
    def __init__(self,path):
        self.terminal = sys.stdout
        self.log = open(path, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def mse_loss(x,y):
    return np.mean(((x.ravel()-y.ravel())*(x.ravel()-y.ravel())))

def mae_loss(x,y):
    return np.mean(np.abs(x.ravel()-y.ravel()))

def psnr(x,y):
    mse = np.mean(((x.ravel()-y.ravel())*(x.ravel()-y.ravel())))
    Q = np.max([x.max(),y.max()])
    return 10*np.log((Q*Q)/mse)