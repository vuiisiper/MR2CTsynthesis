import os
import glob

dir_data= r'D:\PaddedMRCT\Images\Test\MR'

filepaths = glob.glob(os.path.join(dir_data,'*.tif'))
filepath_txt = r'D:\PaddedMRCT\Images\test.txt'
flnames=[]
for path in filepaths:
    flname = os.path.basename(path)
    flnames.append(flname)
with open(filepath_txt,'w') as file:
    for flname in flnames:
        file.write('{}\n'.format(flname))

