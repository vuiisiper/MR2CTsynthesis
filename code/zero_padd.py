import os
import numpy as np
import glob

def zero_padd(nrow,ncol,filepath):
    """
    input image size: 172x220 => zero padded size: 192x220
    then put nrow = 20, ncol = 4
    filepath = 'C:\\Data\\MR_CT_data\\Train\\MR_train\\MR_0002.tif'
    :param nrow
    :param ncol
    :param filepath
    :return: saves the zero padded image in same name same folder
    rana.banik@vanderbilt.edu
    """
    from PIL import Image
    img = Image.open(filepath)
    img = np.array(img)
    [row, col] = img.shape #(172, 220)
    colpad = np.zeros((int(nrow/2), 220))
    zp = np.vstack((colpad,img,colpad)) #(192, 220)
    x = zp.shape[0] #192
    rowpad = np.zeros((x,int(ncol/2)))
    zp = np.hstack((rowpad, zp, rowpad))
    zp = Image.fromarray(zp.astype(dtype=int)) #input(prompt)
    return zp.save(filepath)

#The path is the main folder where branch folders start
path = r'D:\MRI_CT_data\MRCTdata_2D_tifs'
main_path = os.path.join(path)
main_path_norm = os.path.normpath(main_path)
FoldList = os.listdir(main_path)
for each in FoldList:
    path_type = os.path.join(main_path,each)
    ModList = os.listdir(path_type)
    for every in ModList:
        mod_path = os.path.join(path_type,every)
        FileList = glob.glob(os.path.join(mod_path,'*'))
        for ii in FileList:
            zero_padd(20,4,ii)


