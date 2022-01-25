#!/usr/bin/env python
# coding: utf-8

# Function:get all the file names(with the filter) in a directory.
import os
def get_filenames(path,filter_list=['.jpg','.png','.bmp','.tif','.dcm', '.gz', '.nii.gz', '.nrrd']):
    result=[]
    for path,dirlist,filelist in os.walk(path):
        for file in filelist:
            afilepath=os.path.join(path,file)
            ext=os.path.splitext(afilepath)[1]   #os.path.splitext(afilepath)[0] is the file name,os.path.splitext(afilepath)[1] is the file type.
            if filter_list is None:
                result.append(afilepath)
            else:
                if ext in filter_list:
                    result.append(afilepath)
            
    return result

# Function: get all the sub-folders in a folder
import os
def get_subfolder_list(path):
    result=[]
    for path,dirlist,filelist in os.walk(path):
        for directory in dirlist:          
            result.append(os.path.join(path,directory))
            
    return result


# Function: get the list of the first sub-folders in a given dir
from glob import glob
import os.path
def traversalDir_FirstDir(path):
    FirstDir_list = []
    if (os.path.exists(path)):
        files = glob(path + '\\*' )

        for file in files:
            if (os.path.isdir(file)):
                FirstDir_list.append(os.path.split(file)[1])
                
    return FirstDir_list


# Function: make a directory.
def mkdir(path):
    import os
    isExist=os.path.exists(path)
    if not isExist:
        os.makedirs(path) 
        print(path," is created successfully!")
#     else:
#         print(path, "exists already!")  

#Function: convert int16 image to uint8 image.
import numpy as np
def convert_int16_to_uint8(image_int16):
    x_min=image_int16.min()
    x_max=image_int16.max()
    image_uint8=255*((image_int16-x_min)/(x_max-x_min))
    image_uint8=image_uint8.astype(np.uint8)
    
    return image_uint8


    
# Show images
import os
import pydicom
import matplotlib.pyplot as plt
import cv2
def display_dcm_images(image_path_list,title_list=None,image_number_each_row=3,save_path=None):
    num_images=len(image_path_list)
    num_rows=num_images//image_number_each_row+1
      
    plt.figure(figsize=(18,18), dpi=200)
    
    for i in range(num_images):
        #read images
        image_path=image_path_list[i]
        basename=os.path.basename(image_path)
        image_array=pydicom.read_file(image_path).pixel_array
        image_array=convert_int16_to_uint8(image_array)
        
        # plot
        sub_fig=plt.subplot(num_rows,image_number_each_row,i+1)
        title=title_list[i] if title_list!=None else basename
        sub_fig.set_title(title)
        plt.imshow(image_array,'gray')
    
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path,title+'.png'),image_array)

    plt.show()


"""
Function: create a logger to save logs.
"""
import logging
def get_logger(log_file_name):
    logger=logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler=logging.FileHandler(log_file_name)
    handler.setLevel(logging.DEBUG)
    formatter=logging.Formatter('%(asctime)s: %(name)s (%(levelname)s)  %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

        
"""
Define a function to print and save log simultaneously.
"""
def save_log(string, logger=None):
    if logger is None:
        print(string)
    else:
        logger.info(string)
    
def myprint(str, show_logs):
    if show_logs:
        print(str)
        
import pandas as pd    
def transpose_dataframe(df):
    transp_df=pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    
    return transp_df
    

# import numpy as np
# import nrrd
# import nibabel as nib
# def convert_nrrd_to_nii(scr_nrrd_path, dst_nii_path):
#     #load nrrd 
#     _nrrd = nrrd.read(scr_nrrd_path)
#     data = _nrrd[0]
#     header = _nrrd[1]
    
#     #save nifti
#     img = nib.Nifti1Image(data, np.eye(4))
#     nib.save(img, dst_nii_path)
    

import SimpleITK as sitk
def dcm2nii(dcm_path_read, nii_path_save):
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_path_read)
    assert(len(series_id))==1
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_path_read, series_id[0])
    print("series_id={}, len_slices={}.".format(series_id,len(series_file_names)))
    
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3d = series_reader.Execute()
    sitk.WriteImage(image3d, nii_path_save)
    
    
import matplotlib.pyplot as plt
import nibabel as nib
def show_nii_image_slices(nii_image_path):
    image_array=nib.load(nii_image_path).get_fdata()
    for index in range(image_array.shape[2]):
        image_slice=image_array[:,:,index]
        plt.figure(figsize=(5,5))
        plt.title(index, fontsize=12)
        plt.axis('off')
        plt.imshow(image_slice, aspect='auto',cmap='gray')
        plt.show()
        plt.close()
    
    
    
"""
Boolean: if a string starts with one prefix in the prefix_list.
"""    
def startswith(string, prefix_list):
    is_startswith=False
    
    for prefix in prefix_list:
        if string.startswith(prefix):
            is_startswith=True
        
    return is_startswith


from glob import glob
import os
import nibabel as nib


def split_filename(filepath):
    """ split a filepath into the full path, filename, and extension (works with .nii.gz) """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def open_nii(filepath):
    """ open a nifti file with nibabel and return the object """
    image = os.path.abspath(os.path.expanduser(filepath))
    obj = nib.load(image)
    return obj


def save_nii(obj, outfile, data=None, is_nii=False):
    """ save a nifti object """
    if not is_nii:
        if data is None:
            data = obj.get_data()
        nib.Nifti1Image(data, obj.affine, obj.header)\
            .to_filename(outfile)
    else:
        obj.to_filename(outfile)


def glob_nii(dir):
    """ return a sorted list of nifti files for a given directory """
    fns = sorted(glob(os.path.join(dir, '*.nii*')))
    return fns


import json
def save_dict(dict_data, save_txt_path):
    """
    Save a dict to a .txt file
    """
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        str_data = json.dumps(dict_data, ensure_ascii=False) 
        f.write(str_data)
        
def load_dict(txt_path):
    """
    Load the dict from a .txt file.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        dict_data = json.loads(data)

        return dict_data
    
    
    
import pickle
def save_pickle(dict_data, save_pickle_path):
    """
    Save a dict to a .pickle file
    """
    file = open(save_pickle_path, 'wb')
    pickle.dump(dict_data, file)
    file.close()
        
def load_pickle(pickle_path):
    """
    Load the dict from a .pickle file.
    """
    with open(pickle_path, 'rb') as file:
        dict_data =pickle.load(file)

    return dict_data
