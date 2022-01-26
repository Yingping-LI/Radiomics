#!/usr/bin/env pythonhttp://localhost:8888/edit/2020_MRI_Work/Radiogenomics/perform_classification/mySettings.py#
# coding: utf-8
import pandas as pd
import numpy as np
import os


def get_random_seed_list():
    # use random seed 2021 to randomly generate 50 integers as random seeds used in our experiments.
    # see utils/generate_random_number.ipynb for the code about how to generate these random seeds.
    random_seed_list=[1140, 3413, 1152, 2669, 3934, 830, 4765, 2397, 198, 3174, 
                      4166, 4641, 1799, 257,  410, 2992, 3555, 4159, 1585, 2704,
                      4475, 1460, 2680, 3077, 3494, 4494, 3911, 3413, 70, 169,
                      3605, 1270, 1700, 3091, 3257, 1106, 1754, 2959, 552, 1547,
                      403, 1870, 2065, 3250, 1031,  245, 4595, 2284, 4215,  447]
    
    random_seed_list=[2021]+random_seed_list
    
    return random_seed_list

    
def get_basic_settings():
    
    #========================================
    # - "experiment_class": {"BraTS2021",  "TCGA_IDH",  "TCGA_MGMT"}, used to control different datasets for experiments.
    # - "experiment_method": {"binary", "multilabel"}.
    # - "task_name_list": list some tasks in the specified "experiment_class", if it is a empty list, 
    #                     then all the tasks in this experiment_class will be done!
    # - "features_for_TCGA": {"extracted_features", "public_features"}, use public features from TCIA, or use features extracted by myself.
    # - "normalization_method": "no_normalization", #{"no_normalization", "fcm", "zscore"} 
    # - "feature_filter_dict": used to control the settings to filter the features for classification.
    #                     - "modality_list": ["t1", "t1ce", "t1Gd", "t2", "flair"];
    #                     - "imageType_list": ["original", "gradient", "log-sigma-1-0-mm-3D", "log-sigma-3-0-mm-3D",
    #                         "square", "squareroot", "logarithm", "exponential", "lbp-3D-m2", "lbp-3D-m1", "lbp-3D-k",
    #                         "wavelet-LLH", "wavelet-LHL", "wavelet-LHH", "wavelet-HLL", "wavelet-HLH", "wavelet-HHL", "wavelet-HHH", "wavelet-LLL",],
    #                     - "tumor_subregion_list": ["NCR", "ED", "ET", "TC", "wholeTumor"]
    # - "feature_selection_method": {"RFECV","RFE", AnovaTest, SelectFromModel, PCA, ChiSquare, MutualInformation}
    #                     Note that: AnovaTest is very fast and effective.
    # - "imbalanced_data_strategy":  {"SMOTE", "BorderlineSMOTE", "SVMSMOTE", "RandomOverSampler", "RandomUnderSampler", "SMOTE_RandomUnderSampler", "IgnoreDataImbalance"}
    # - "harmonization_method": {"withoutComBat", "parametric_ComBat", "nonParametric_ComBat, "noEB_ComBat"}
    # - "harmonization_label": {"Tissue.source.site", "is_3T", "is_3T_mostCommon"}, column name of the setting label used to do the harmonization.
    # - "random_seed": int number, used for reproducibility of the results.

    random_seed_list=get_random_seed_list()
    
    '''
    Basic Settings for the code
    '''
    global_basic_settings={
        "experiment_class": "TCGA_IDH",
        "experiment_method": "binary", #"multilabel",
        "task_list": ["TCGA_1.104.02_isGBM_withAge"], 
    #    "task_list": ["TCGA_2.106_isIDHMutant_CC-withPredictLable"],
    #    "task_list": ["TCGA_3.106_is1p19qCodeleted_CC-withPredictLable"], 
        "features_for_TCGA": "extracted_features",  
        "normalization_method": "zscore", #{"no_normalization", "fcm", "zscore"} 
        "feature_filter_dict":{"modality_list": ["t1", "t1ce", "t1Gd", "t2", "flair"], 
                             "imageType_list": ["original"],  # "original" for predicting tumor grade, "squareroot" for predicting IDH status, "log-sigma-1-0-mm-3D" for predicting 1p/19q.
                             "tumor_subregion_list": ["NCR", "TC", "wholeTumor"], #["NCR", "ED", "ET", "TC", "wholeTumor"],
                            },
        "feature_selection_method":"AnovaTest",
        "imbalanced_data_strategy": "IgnoreDataImbalance", 
        #---settings for harmonization---
        "harmonization_settings": {
            "harmonization_method": "withoutComBat",
            "ComBat_batch_col": "is_3T_t1", #{"Tissue.source.site", #"is_3T_t1",}
            "ComBat_categorical_cols": None, #{None, ["is_female"],}
            "ComBat_continuous_cols":  None, #{None, ["age"]}
            "ComBat_ref_batch": None, # 1, "Henry Ford Hospital"
        },
        #--random seed---
        "random_seed_list": random_seed_list, #[2021],
    }

   
    return global_basic_settings


