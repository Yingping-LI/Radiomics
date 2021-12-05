#!/usr/bin/env python
# coding: utf-8

import os

def get_arrange_results_settings_dict():
    """
    Settings used to arrange and plot the results;
    """
    basepath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/Results_randomseed2021"
    
    arrange_results_settings_dict={}

    #==================== 1: Compare the normalization method =====================
    arrange_results_settings_dict["compare_normalization_methods"]={
        "results_basepath": os.path.join(basepath, "1-compare_normalization_methods"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "normalization_method",
                         "rename_hue_values": {"no_normalization": "Without normalization", 
                                               "zscore": "With Z-Score"},
                         "ncol": 2,
                         "exclude_hue_value": ["fcm"]
                        }
    }
    
    
    #====================== 2: Compare different image feature extraction strategy ===================================
    arrange_results_settings_dict["compare_feature_strategy"]={
        "results_basepath": os.path.join(basepath, "2-compare_feature_strategy"),
        "groupby_column": "base_task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "task_additional_description",
                         "rename_hue_values": {"WT base": "WT base ", 
                                               "WT withSubregionInfo":"WT withIndicatorColumns ", 
                                               "NCR-TC-WT base":"NCR-TC-WT base", 
                                               "NCR-TC-WT withSubregionInfo": "NCR-TC-WT withIndicatorColumns", 
                                               "NCR-TC-WT-ED-ET base": "NCR-TC-WT-ED-ET base", 
                                               "NCR-TC-WT-ED-ET withSubregionInfo":"NCR-TC-WT-ED-ET withIndicatorColumns"},
                         "ncol": 3,
                         "exclude_hue_value": ["ShapeFeatureOnly base", "ShapeFeatureOnly withIndicatorColumns"]
                        }
    }
     
    #====================== 3: Compare different image filters ===================================
    arrange_results_settings_dict["compare_image_filter"]={
        "results_basepath": os.path.join(basepath, "3-compare_image_filter"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "image_filter",
                         "rename_hue_values": {"exponential": "Exponential", 
                                               "square": "Square", 
                                               "lbp-3D": "Local Binary Pattern",  
                                               "gradient": "Gradient", 
                                               "wavelet": "Wavelet", 
                                               "original": "Original", 
                                               "squareroot": "SquareRoot", 
                                               "logarithm": "Logarithm", 
                                               "log-sigma-1-0-mm-3D": "Laplacian of Gaussian"},
                         "ncol": 5,
                         "exclude_hue_value": ["log-sigma-3-0-mm-3D"]
                        }
    }
    
    #====================== 4: Compare whether to add clinical info (age and sex) ===================================
    arrange_results_settings_dict["compare_add_clinicalinfo"]={
        "results_basepath": os.path.join(basepath, "4-compare_add_clinicalinfo"),
        "groupby_column": "base_task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "task_additional_description",
                         "rename_hue_values":{" withSubregionInfo": "Without clinical info",
                                          " withAllInfo": "With clinical info"},
                         "ncol": 2,
                         "exclude_hue_value": []
                        }
    }
    return arrange_results_settings_dict


