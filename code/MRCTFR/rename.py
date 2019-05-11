# D:\PaddedMRCT\Images\Test\MR
import os
import glob
from shutil import move
from tqdm import tqdm
# dir_data= r'D:\PaddedMRCT\Images\Test\MR'
# dir_data = r'D:\PaddedMRCT\Images\Test\CT\phase'
dir_data= r'D:\PaddedMRCT\Images\Valid\CT\phase'

filepaths = glob.glob(os.path.join(dir_data,'*.tif'))

for path in tqdm(filepaths):
    path_dest = path.replace('CT_','')
    path_dest = path_dest.replace('_phase','')
    # print(path, path_dest)
    move(path,path_dest)

