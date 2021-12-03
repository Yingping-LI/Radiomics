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
                         "hue_order": ["no_normalization", "zscore"],
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
                         "hue_order": ["WT_base", "WT_withIndicatorColumns", "NCR-TC-WT_base", "NCR-TC-WT_withIndicatorColumns", "NCR-TC-WT-ED-ET_base", "NCR-TC-WT-ED-ET_withIndicatorColumns"],
                         "ncol": 3,
                         "exclude_hue_value": ["ShapeFeatureOnly_base", "ShapeFeatureOnly_withIndicatorColumns"]
                        }
    }
    
    
    #====================== 3: Compare different image filters ===================================
    arrange_results_settings_dict["compare_image_filter"]={
        "results_basepath": os.path.join(basepath, "3-compare_image_filter"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "image_filter",
                         "hue_order": ["exponential", "square", "lbp-3D",  "gradient", "wavelet", "original", "squareroot", "logarithm", "log-sigma-1-0-mm-3D"],
                         "ncol": 5,
                         "exclude_hue_value": ["log-sigma-3-0-mm-3D"]
                        }
    }
    
    return arrange_results_settings_dict





