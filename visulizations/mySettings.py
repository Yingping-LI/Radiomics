#!/usr/bin/env python
# coding: utf-8

import os

def get_image_visualization_setting_dict():
    # base path
    base_dataPath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    
    # base settings
    modality_list=["t1", "t1Gd", "t2", "flair"]
    normalization_method="zscore" #{"no_normalization", "fcm", "zscore"} 
        
       
    #define settings for visualizing images.
    image_visualization_setting_dict={}
    image_visualization_setting_dict["TCGA_train"]={
        "image_folder": base_dataPath+"/originalData/TCGA/TCIA_Segmentation/TCGA_arranged/TCGA_train/Images",
        "segmentation_folder": base_dataPath+"/originalData/TCGA/TCIA_Segmentation/TCGA_arranged/TCGA_train/segmentation",
    }
        
    image_visualization_setting_dict["TCGA_test"]={
        "image_folder": base_dataPath+"/originalData/TCGA/TCIA_Segmentation/TCGA_arranged/TCGA_test/Images",
        "segmentation_folder": base_dataPath+"/originalData/TCGA/TCIA_Segmentation/TCGA_arranged/TCGA_test/segmentation",
    }

    ##=============== Add other distributions =============================
    for setting_name, image_visualization_setting in image_visualization_setting_dict.items():
        image_visualization_setting["normalization_method"]=normalization_method
        image_visualization_setting["modality_list"]=modality_list
        if normalization_method!="no_normalization":
            image_visualization_setting["image_folder"]=os.path.join(base_dataPath, "PreprocessedImages", setting_name, normalization_method, normalization_method+"_normalizedImages")
        image_visualization_setting_dict[setting_name]=image_visualization_setting   
        
    return image_visualization_setting_dict
