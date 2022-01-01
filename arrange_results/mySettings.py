#!/usr/bin/env python
# coding: utf-8

import os

#=================  used for "main_arrange_results.ipynb" ==================
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
     
    ====================== 3: Compare different image filters ===================================
    #---- Compare Wavelet image filter ----
    arrange_results_settings_dict["compare_image_filter-wavelet"]={
        "results_basepath": os.path.join(basepath, "3-compare_image_filter"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "image_filter",
                         "rename_hue_values": {
                                               "wavelet-HHH": "wavelet-HHH", 
                                               "wavelet-HHL": "wavelet-HHL", 
                                               "wavelet-HLH": "wavelet-HLH", 
                                               "wavelet-HLL": "wavelet-HLL", 
                                               "wavelet-LHH": "wavelet-LHH", 
                                               "wavelet-LHL": "wavelet-LHL", 
                                               "wavelet-LLH": "wavelet-LLH", 
                                               "wavelet-LLL": "wavelet-LLL", 
                                               },
                         "ncol": 4,
                         "exclude_hue_value": ["exponential", "square", "lbp-3D-m1", "lbp-3D-m2", "lbp-3D-k", "gradient", "original",
                                              "squareroot", "logarithm", "log-sigma-1-0-mm-3D", "log-sigma-3-0-mm-3D"]
                        }
    }
    
    arrange_results_settings_dict["compare_image_filter"]={
        "results_basepath": os.path.join(basepath, "3-compare_image_filter"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "image_filter",
                         "rename_hue_values": {
                                               "square": "Square", 
                                               "exponential": "Exponential", 
                                               "squareroot": "SquareRoot", 
                                               "logarithm": "Logarithm", 
                                               "lbp-3D-m1": "Local Binary Pattern", 
                                               "wavelet-LLL": "Wavelet-LLL", 
                                               "log-sigma-1-0-mm-3D": "Laplacian of Gaussian",
                                               "gradient": "Gradient",
                                               "original": "Original", },
                         "ncol": 5,
                         "exclude_hue_value": ["wavelet-HHH", "wavelet-HHL", "wavelet-HLH", "wavelet-HLL",
                                              "wavelet-LHH", "wavelet-LHL", "wavelet-LLH",
                                              "log-sigma-3-0-mm-3D", "lbp-3D-m2", "lbp-3D-k"]
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
    
    
   #====================== 5: Compare the data imbalance strategy ===================================
    arrange_results_settings_dict["compare_data_imbalance"]={
        "results_basepath": os.path.join(basepath, "5-compare_dataimbalance"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "Data_imblance_strategy",
                         "rename_hue_values":{"IgnoreDataImbalance": "Without data imbalance strategy",
                                              "RandomOverSampler": "RandomOverSampler",
                                              "RandomUnderSampler": "RandomUnderSampler",
                                              "SMOTE": "SMOTE",
                                              "SVMSMOTE": "SVMSMOTE",
                                              "BorderlineSMOTE": "BorderlineSMOTE",
                                              "SMOTE_RandomUnderSampler": "SMOTE and RandomUnderSampler"},
                         "ncol": 4,
                         "exclude_hue_value": []
                        }
    }
    
    #====================== 6: Compare ComBat ===================================
    arrange_results_settings_dict["compare_ComBat"]={
        "results_basepath": os.path.join(basepath, "6-compare_ComBat"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "ComBat_method",
                         "rename_hue_values":{
                             "noEB_ComBat_Tissue.source.site_noCovars": "Standard ComBat (site)",
                             "noEB_ComBat_Tissue.source.site_withCovars": "Standard ComBat (site, covariates=[age, sex])",
                             "noEB_ComBat_is_3T_t1_noCovars": "Standard ComBat (is_3T)",
                             "noEB_ComBat_is_3T_t1_withCovars": "Standard ComBat (is_3T,  covariates=[age, sex])",
                             "nonParametric_ComBat_Tissue.source.site_noCovars": "Non-parametric ComBat (site)",
                             "nonParametric_ComBat_Tissue.source.site_withCovars": "Non-parametric ComBat (site,  covariates=[age, sex])",
                             "nonParametric_ComBat_is_3T_t1_noCovars": "Non-parametric ComBat (is_3T)",
                             "nonParametric_ComBat_is_3T_t1_withCovars": "Non-parametric ComBat (is_3T,  covariates=[age, sex])",
                             "parametric_ComBat_Tissue.source.site_noCovars": "Parametric ComBat (site)",
                             "parametric_ComBat_Tissue.source.site_withCovars": "Parametric ComBat (site,  covariates=[age, sex])",
                             "parametric_ComBat_is_3T_t1_noCovars": "Parametric ComBat (is_3T)",
                             "parametric_ComBat_is_3T_t1_withCovars": "Parametric ComBat (is_3T,  covariates=[age, sex])",
                             "withoutComBat": "without ComBat"},
                         "ncol": 3,
                         "exclude_hue_value": []
                        }
    }
    
    #====================== 7: compare using classifier chain and adding the true GBM and IDH label ======================
    arrange_results_settings_dict["compare_CC_truelabel"]={
        "results_basepath": os.path.join(basepath, "7-compare_CC_truelabel"),
        "groupby_column": "base_task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "task_additional_description",
                         "rename_hue_values": {" withSubregionInfo":"Without GBM and IDH true labels", 
                                              " CC-withTrueLable": "with GBM and IDH true labels"},
                         "ncol": 2,
                         "exclude_hue_value": []
                        }
    }
    
    #====================== 8: compare using classifier chain and adding the predicted GBM and IDH label ======================
    arrange_results_settings_dict["compare_ClassifierChain"]={
        "results_basepath": os.path.join(basepath, "7-compare_ClassifierChain"),
        "groupby_column": "base_task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "task_additional_description",
                         "rename_hue_values": {
                             " withSubregionInfo":"Without GBM and IDH labels", 
                             " CC-withPredictLable": "with predicted GBM and IDH labels",
                             " CC-withTrueLable": "with true GBM and IDH labels"
                         },
                         "ncol": 3,
                         "exclude_hue_value": []
                        }
    }
    return arrange_results_settings_dict





#================ used for "main_transform_binary-multiclass_classification.ipynb" ======================
"""
Settings for converting three binary classification results to one multiclass classification results;
"""
def get_convert_binary_to_multiclass_setting_dict():
    
    # base path
    base_dataPath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    image_filter="log-sigma-1-0-mm-3D"
    intensity_normalization="zscore"
    random_seed=0
    
    results_base_path= os.path.join(base_dataPath, "Results_randomseed"+str(random_seed), image_filter, "TCGA_IDH-extracted_features-"+intensity_normalization, "withoutComBat-IgnoreDataImbalance")
    
    convert_binary_to_multiclass_setting_dict={}
    
    convert_binary_to_multiclass_setting_dict["TCGA-IDH"]={
        # folders of the tasks.
        "binary_task_path_dict": {"is_GBM": os.path.join(results_base_path, "TCGA_1.103_isGBM_withSubregionInfo"),
                                  "is_IDH_mutant": os.path.join(results_base_path, "TCGA_2.106_isIDHMutant_CC-withPredictLable"),
                                  "is_1p19q_codeleted": os.path.join(results_base_path, "TCGA_3.106_is1p19qCodeleted_CC-withPredictLable")},
        # base path to save the results.
        "save_results_basepath": results_base_path, 
        # excel path which saves the ground truth labels;
        "ground_truth_target_excel_dict": {"train_data": os.path.join(base_dataPath, "Features", "final_metadata", intensity_normalization, "TCGA_extracted_features_IDH_train_resplited_randomseed_"+str(random_seed)+".xlsx"),
                                           "test_data": os.path.join(base_dataPath, "Features", "final_metadata", intensity_normalization, "TCGA_extracted_features_IDH_test_resplited_randomseed_"+str(random_seed)+".xlsx")
                                          }
                                           

    }
        
    return convert_binary_to_multiclass_setting_dict

