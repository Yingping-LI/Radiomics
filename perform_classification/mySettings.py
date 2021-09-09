#!/usr/bin/env pythonhttp://localhost:8888/edit/2020_MRI_Work/Radiogenomics/perform_classification/mySettings.py#
# coding: utf-8
import pandas as pd
import os

#========================================
# - "experiment_class": {"BraTS2021",  "TCGA_IDH",  "TCGA_MGMT"}, used to control different datasets for experiments.
# - "task_name_list": list some tasks in the specified "experiment_class", if it is a empty list, 
#                     then all the tasks in this experiment_class will be done!

'''
Basic Settings for the code
'''
global_basic_settings={
    "experiment_class": "BraTS2021", #"BraTS2021",  "TCGA_IDH",  "TCGA_MGMT"
    "task_list": [], #["TCGA-LGG_3.201_is1p19qCodeleted_base", "TCGA-LGG_3.202_is1p19qCodeleted_with_clinicalInfo"],  
    "features_for_TCGA": "extracted_features", #"extracted_features", "public_features"
    "feature_filter_dict":{"modality_list": ["t1", "t1ce", "t2", "flair"], #["t1", "t1Gd", "t2", "flair"], ["t1", "t1ce", "t2", "flair"],
                         "imageType_list": ["gradient"],
                                          #["original", "gradient", "log-sigma-1-0-mm-3D", "log-sigma-3-0-mm-3D"],
                         "tumor_subregion_list": ["NCR", "ED", "ET", "TC", "wholeTumor"], #["NCR", "ED", "ET", "TC", "wholeTumor"]
                        },
    "feature_selection_method":"AnovaTest", #"RFECV","RFE", AnovaTest, SelectFromModel
    "use_randomSearchCV":True, #False, True
    "harmonization_method": "withoutComBat", # withoutComBat, "parametric_ComBat", nonParametric_ComBat, noEB_ComBat
    "harmonization_label": "is_3T",      #"Tissue.source.site", "is_3T"
    "random_seed": 2021,
}


def get_basic_settings():
   
    return global_basic_settings
