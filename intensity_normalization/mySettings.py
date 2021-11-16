#!/usr/bin/env python
# coding: utf-8

import os

"""
Settings for intensity normalization;
"""
def get_intensity_normalization_setting_dict():
    
    # base settings and base image path
    normalization_method="fcm" #{"fcm", "zscore"} 
    modality_list=["t1", "t1Gd", "t2", "flair"] 
    base_dataPath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    save_normalized_image_basepath=base_dataPath+"/PreprocessedImages"     
       
    #define settings for each feature extraction.
    intensity_normalization_setting_dict={}
    intensity_normalization_setting_dict["TCGA_train"]={
        "image_folder": base_dataPath+"/originalData/TCGA/TCIA_Segmentation/TCGA_arranged/TCGA_train/Images",
        #"image_folder": base_dataPath+"/originalData/temp_test",
    }
        
    intensity_normalization_setting_dict["TCGA_test"]={
        "image_folder": base_dataPath+"/originalData/TCGA/TCIA_Segmentation/TCGA_arranged/TCGA_test/Images",
    }
    
    ##=============== Add other distributions =============================
    for setting_name, intensity_normalization_setting in intensity_normalization_setting_dict.items():
        intensity_normalization_setting["normalization_method"]=normalization_method
        intensity_normalization_setting["modality_list"]=modality_list
        intensity_normalization_setting["normalized_image_basepath"]=os.path.join(save_normalized_image_basepath, setting_name)
        intensity_normalization_setting_dict[setting_name]=intensity_normalization_setting
        
    return intensity_normalization_setting_dict


