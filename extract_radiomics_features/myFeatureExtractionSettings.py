#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("E://2020_MRI_Work/HarmonizationProject")
from utils.myUtils import mkdir

#possible values for the parameters:
#    - "extractor_setting_file": "None" means using default settings.
#                              "./feature_extract_setting.yaml",
#    - "binCount": 'binCount_32','binCount_64','binCount_N' with any int N.

global_basic_settings={
    "extractor_setting_file": "./feature_extract_setting.yaml",  # None
    "binCount":'binCount_64',
    "feature_filter": ['firstorder', 'glcm','glrlm', 'shape', 'glszm','gldm','ngtdm'],
    "label_dict": {"NCR": 1, 
                "ED": 2,
                "ET": 4,
                "TC": [1,4],
                "wholeTumor": [1, 2, 4]},
}


def get_basic_settings():
   
    return global_basic_settings


"""
Feature extractor setting for BraTS2021 competition data.
"""
def get_feature_extract_setting_dict1():
    # base path
    base_dataPath="G://DURING PHD/5)Glioblastoma_MGMT_RSNA-MICCAI"
    feature_save_folder=base_dataPath+"/Features/extracted_features"
    mkdir(feature_save_folder)
    
    # base settings
    modality_list=["t1", "t1ce", "t2", "flair"]
    normalization_method=None  # None, "zscore"
        
       
    #define settings for each feature extraction.
    feature_extract_setting_dict={}
    
    feature_extract_setting_dict["BraTS2021_MGMT_train"]={
        "image_folder": base_dataPath+"/BraTS2021_Segmentation_filtered/BraTS2021_TrainingData",
        "segmentation_folder": base_dataPath+"/Predict_Segmentation/imagesTr_GT_filtered",     
    }
        
    feature_extract_setting_dict["BraTS2021_MGMT_validation"]={
        "image_folder": base_dataPath+"/BraTS2021_Segmentation_filtered/BraTS2021_ValidationData",
        "segmentation_folder": base_dataPath+"/Predict_Segmentation/imagesVal_converted_filtered",
    }
        
    feature_extract_setting_dict["BraTS2021_MGMT_train_dcm_to_nii"]={
        "image_folder": base_dataPath+"/MGMT_classification_nii/train/3_registered_nii",
        "segmentation_folder": base_dataPath+"/Predict_Segmentation/imagesTr_dcm_to_nii_converted",
    }
    
    feature_extract_setting_dict["BraTS2021_MGMT_validation_dcm_to_nii"]={
        "image_folder": base_dataPath+"/MGMT_classification_nii/validation/3_registered_nii",
        "segmentation_folder": base_dataPath+"/Predict_Segmentation/imagesVal_dcm_to_nii_converted",
    }
    
    
    ##=============== Add other distributions =============================
    for setting_name, feature_extract_setting in feature_extract_setting_dict.items():
        feature_extract_setting["modality_list"]=modality_list
        feature_extract_setting["label_dict"]=get_basic_settings()["label_dict"]
        
        if normalization_method is not None:
            feature_extract_setting["image_folder"]=feature_extract_setting["image_folder"]+"_"+normalization_method
            feature_extract_setting["save_excel_path"]=feature_save_folder+"/features_"+setting_name+"_"+normalization_method+".xlsx"
        else:
            feature_extract_setting["save_excel_path"]=feature_save_folder+"/features_"+setting_name+".xlsx" 
         
        feature_extract_setting_dict[setting_name]=feature_extract_setting
        
    return feature_extract_setting_dict



"""
Feature extractor setting for TCGA-dataset.
"""
def get_feature_extract_setting_dict():
    # base path
    base_dataPath="C://YingpingLI/Glioma/TCGA"
    feature_save_folder=base_dataPath+"/Features/extracted_features"
    mkdir(feature_save_folder)
    
    # base settings
    modality_list=["t1", "t1Gd", "t2", "flair"]
    normalization_method=None  # None, "zscore"
        
       
    #define settings for each feature extraction.
    feature_extract_setting_dict={}
    
    feature_extract_setting_dict["TCGA_train"]={
        "image_folder": base_dataPath+"/TCIA_Segmentation/TCGA_arranged/TCGA_train/Images",
        "segmentation_folder": base_dataPath+"/TCIA_Segmentation/TCGA_arranged/TCGA_train/segmentation",
    }
        
    feature_extract_setting_dict["TCGA_test"]={
        "image_folder": base_dataPath+"/TCIA_Segmentation/TCGA_arranged/TCGA_test/Images",
        "segmentation_folder": base_dataPath+"/TCIA_Segmentation/TCGA_arranged/TCGA_test/segmentation",
    }
    
    
    ##=============== Add other distributions =============================
    for setting_name, feature_extract_setting in feature_extract_setting_dict.items():
        feature_extract_setting["modality_list"]=modality_list
        feature_extract_setting["label_dict"]=get_basic_settings()["label_dict"]
        
        if normalization_method is not None:
            feature_extract_setting["image_folder"]=feature_extract_setting["image_folder"]+"_"+normalization_method
            feature_extract_setting["save_excel_path"]=feature_save_folder+"/features_"+setting_name+"_"+normalization_method+".xlsx"
        else:
            feature_extract_setting["save_excel_path"]=feature_save_folder+"/features_"+setting_name+".xlsx" 
         
        feature_extract_setting_dict[setting_name]=feature_extract_setting
        
    return feature_extract_setting_dict