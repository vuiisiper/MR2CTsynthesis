import os
import glob
import nibabel as nib
from PIL import Image
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt

def argo():
    parser = argparse.ArgumentParser(description='split 3d image into multiple 2d images')
    parser.add_argument('img_dir', type=str,help='path to nifti image directory')  #parser.add_argument('img_dir', type=str)
    parser.add_argument('out_dir', type=str,help='path to tif image directory')
    parser.add_argument('-a', '--axis', type=int, default=2,help='axis of 3D images on which the slices will be taken') #putting dashes and double dashes making the argument optional, not required
    parser.add_argument('-p', '--pct-range', nargs=2, type=float, default=(0.0,1.0),help=('range of indices, as a percentage, from which to sample ' 
                              'in each 3d image volume. used to avoid creating blank tif '
                              'images if there is substantial empty space along the ends '
                              'of the chosen axis'))
    return parser

def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    # File_path = os.path.join(path, base)
    return path,base,ext
##
def main():
    try:
        args = argo().parse_args()
        fns = glob.glob(os.path.join(args.Subjects,'*.nii*'))
        for fn in fns:
            path, base, _ = split_filename(fn) #just taking the base, others don't care #before _,base,_
            img = nib.load(fn).get_data()
            start = int(args.pct_range[0] * img.shape[args.axis])
            end = int(args.pct_range[1] * img.shape[args.axis]) + 1
            for i in range(start, end):
                I = Image.fromarray(img[i,:,:]) if args.axis == 0 else \
                    Image.fromarray(img[:,i,:]) if args.axis == 1 else \
                    Image.fromarray(img[:,:,i])
                I.save(os.path.join(args.out_dir,f'{base}_{i}.tif'))
        return 0
    except Exception as e:
        print(e)
        return 1
##
data_dir = os.path.join('D:\\','MRI_CT_data') #D:\MRI_CT_data #forward slashes for Linux but also works in Windows
FolderList = os.listdir(data_dir)
# Folders = os.path.join()
for each in FolderList:
    Subjects = os.path.join(data_dir,each)
    LoadFile = os.path.join(Subjects,'*.nii')  #if I take '*.nii*' it takes all the file both .gz and .nii
    FileList = glob.glob(os.path.join(Subjects,'*.nii'))
    Current_fd = Path(Subjects)  # Path module converts to windows path format?
    Current_path = Current_fd / 'MRI_TIF'  # defining the name of the next folder
    Current_path.mkdir(parents=True, exist_ok=True)  # creating the folder

    # for im in range(len(FileList)-1):
    #     Img = nib.load(FileList[im]).get_data()
    #     ax = Img.shape[2]
    #     for i in range(ax):
    #         I = Image.fromarray()


# print(os.getcwd())
# DATA_PATH = Path("data")  # কোড বর্তমানে যে ফোল্ডারে আছে সে ফোল্ডারে নতুন ফোল্ডারের নাম ঠিক করলাম
# PATH = DATA_PATH / each  # এ নামের সাবফোল্ডারের জন্য নাম ঠিক করলাম
# PATH.mkdir(parents=True,exist_ok=True)

if __name__=='__main__':
    sys.exit()



