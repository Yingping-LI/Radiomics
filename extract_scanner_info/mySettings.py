#!/usr/bin/env python
# coding: utf-8
import os

def get_scanner_info_extraction_setting_dict():
    ## TCGA data.
    scanner_info_extraction_setting_dict=get_scanner_info_extraction_setting_dict_TCGA()
    
    ## BraTS2021 data.
    #scanner_info_extraction_setting_dict=get_scanner_info_extraction_setting_dict_BraTs2021()
    
    return scanner_info_extraction_setting_dict


#==================================================
"""
Scanner info extraction settings for TCGA dataset.
"""
def get_scanner_info_extraction_setting_dict_TCGA():
    
    base_path="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    base_results_path=base_path+"/Features/scanner_info"

    scanner_info_extraction_setting_dict={}
    scanner_info_extraction_setting_dict["TCGA_GBM"]={
        "description": "TCGA",
        "dcm_image_folder": base_path+"/originalData/TCGA/TCIA_Segmentation/TCGA-GBM/Images/TCGA-GBM",
        "base_results_path": base_results_path,
        "lgg_gbm_class": "TCGA-GBM"
    }
    
    scanner_info_extraction_setting_dict["TCGA_LGG"]={
        "description": "TCGA",
        "dcm_image_folder": base_path+"/originalData/TCGA/TCIA_Segmentation/TCGA-LGG/Images/TCGA-LGG",
        "base_results_path": base_results_path,
        "lgg_gbm_class": "TCGA-LGG"
    }

    return scanner_info_extraction_setting_dict


"""
Scanner info extraction settings for BraTS2021 dataset.
"""
def get_scanner_info_extraction_setting_dict_BraTs2021():
    
    base_path="G://PhDProjects/RadiogenomicsProjects/BraTS2021"   
    base_results_path=base_path+"/Features/scanner_info"

    modality_dict={"T1w": "t1", 
                   "T1wCE": "t1ce", 
                   "T2w": "t2",
                   "FLAIR": "flair"}
    
    scanner_info_extraction_setting_dict={}
    scanner_info_extraction_setting_dict["BraTS2021_train"]={
        "description": "BraTs2021",
        "dcm_image_folder": base_path+"/originalData/MGMT_classification/train",
        "base_results_path": base_results_path,
        "modality_dict": modality_dict
    }
    
    scanner_info_extraction_setting_dict["BraTS2021_validation"]={
        "description": "BraTs2021",
        "dcm_image_folder": base_path+"/originalData/MGMT_classification/validation",
        "base_results_path": base_results_path,
        "modality_dict": modality_dict
    }

    return scanner_info_extraction_setting_dict