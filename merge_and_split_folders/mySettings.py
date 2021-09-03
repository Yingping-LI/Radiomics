#!/usr/bin/env python
# coding: utf-8

"""
Settings for folder split!
"""
def get_split_folder_setting_dict():
    # base path
    base_dataPath="C://YingpingLI/Glioma/TCGA/TCIA_Segmentation"     
       
    #define settings.
    split_folder_setting_dict={}
    
    split_folder_setting_dict["TCGA_train"]={
        "image_folder": base_dataPath+"/TCGA/Pre-operative_TCGA_NIfTI_and_Segmentations",
        "save_basepath": base_dataPath+"/TCGA_arranged/TCGA_train",
        "total_number_of_modalities":6,
        "used_to_find_basename": "flair",
        "split_criterion": {"Images":["t1", "t1Gd", "t2", "flair"],
                           "segmentation":["GlistrBoost_ManuallyCorrected"]},
    }
        
    split_folder_setting_dict["TCGA_test"]={
        "image_folder": base_dataPath+"/TCGA/Pre-operative_TCGA_NIfTI_and_Segmentations_(BraTS2017TestingData)",
        "save_basepath": base_dataPath+"/TCGA_arranged/TCGA_test",
        "total_number_of_modalities":6,
        "used_to_find_basename": "flair",
        "split_criterion": {"Images":["t1", "t1Gd", "t2", "flair"],
                           "segmentation":["GlistrBoost_ManuallyCorrected"]},
    }
  
        
    return split_folder_setting_dict



