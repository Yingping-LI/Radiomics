#!/usr/bin/env python
# coding: utf-8

import os

def get_arrange_results_settings_dict():
    """
    Settings used to arrange and plot the results;
    """
    basepath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/Results_randomseed2021"
    
    arrange_results_settings_dict={}
    
    arrange_results_settings_dict["compare_normalization_methods"]={
        "results_basepath": os.path.join(basepath, "1-compare_normalization_methods"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "normalization_method",
                         "hue_order": ["no_normalization", "zscore", "fcm"]}
    }
    
    return arrange_results_settings_dict





