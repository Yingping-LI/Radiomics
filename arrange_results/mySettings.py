#!/usr/bin/env python
# coding: utf-8

import os

#=================  used for "main_arrange_results.ipynb" ==================
def get_arrange_results_settings_dict():
    """
    Settings used to arrange and plot the results;
    """
    basepath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/1_Results_NCR-TC-WT-LoGfilter/Results_randomseed2021"
    
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
                         "rename_hue_values": {"WT base": "WT", 
                                               "WT withSubregionInfo":"WT with indicator columns", 
                                               "NCR-TC-WT base":"NCR-TC-WT", 
                                               "NCR-TC-WT withSubregionInfo": "NCR-TC-WT with indicator columns", 
                                               #"NCR-TC-WT-ED-ET base": "NCR-TC-WT-ED-ET base", 
                                               #"NCR-TC-WT-ED-ET withSubregionInfo":"NCR-TC-WT-ED-ET withIndicatorColumns"
                                              },
                         "ncol": 4,
                         "exclude_hue_value": ["ShapeFeatureOnly base", "ShapeFeatureOnly withIndicatorColumns", "NCR-TC-WT-ED-ET base", "NCR-TC-WT-ED-ET withSubregionInfo"]
                        }
    }
     
    #====================== 3: Compare different image filters ===================================
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
                                               "gradient": "Gradient",
                                               "wavelet-LLL": "Wavelet-LLL", 
                                               "log-sigma-1-0-mm-3D": "Laplacian of Gaussian",
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
                         "rename_hue_values":{" withSubregionInfo": "Without age and sex",
                                              " withAge": "With age",
                                              " withSex": "With sex",
                                              " withAllInfo": "With age and sex"},
                         "ncol": 4,
                         "exclude_hue_value": []
                        }
    }
    
    
   #====================== 5: Compare the data imbalance strategy ===================================
    arrange_results_settings_dict["compare_data_imbalance"]={
        "results_basepath": os.path.join(basepath, "6-compare_dataimbalance"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "Data_imblance_strategy",
                         "rename_hue_values":{"IgnoreDataImbalance": "base",
                                              "IgnoreDataImbalance_WithBalacedWeighting": "with balanced weights",
                                              "RandomOverSampler": "Random over-sampling",
                                              "RandomUnderSampler": "Random under-sampling",
                                              "SMOTE": "SMOTE",
                                              "SVMSMOTE": "SVM SMOTE",
                                              "BorderlineSMOTE": "Borderline SMOTE",
                                              "SMOTE_RandomUnderSampler": "SMOTE and random under-sampling"},
                         "ncol": 4,
                         "exclude_hue_value": []
                        }
    }
    
    #====================== 6: Compare ComBat ===================================
    arrange_results_settings_dict["compare_ComBat"]={
        "results_basepath": os.path.join(basepath, "5-compare_ComBat"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "ComBat_method",
                         "rename_hue_values":{
                             "noEB_ComBat_Tissue.source.site_noCovars": "Standard ComBat (site)",
                             "noEB_ComBat_Tissue.source.site_withCovars": "Standard ComBat (site with covariates)",
                             "noEB_ComBat_is_3T_t1_noCovars": "Standard ComBat (3T)",
                             "noEB_ComBat_is_3T_t1_withCovars": "Standard ComBat (3T with covariates)",
                             "nonParametric_ComBat_Tissue.source.site_noCovars": "Nonparametric ComBat (site)",
                             "nonParametric_ComBat_Tissue.source.site_withCovars": "Nonparametric ComBat (site with covariates)",
                             "nonParametric_ComBat_is_3T_t1_noCovars": "Nonparametric ComBat (3T)",
                             "nonParametric_ComBat_is_3T_t1_withCovars": "Nonparametric ComBat (3T with covariates)",
                             "parametric_ComBat_Tissue.source.site_noCovars": "Parametric ComBat (site)",
                             "parametric_ComBat_Tissue.source.site_withCovars": "Parametric ComBat (site with covariates)",
                             "parametric_ComBat_is_3T_t1_noCovars": "Parametric ComBat (3T)",
                             "parametric_ComBat_is_3T_t1_withCovars": "Parametric ComBat (3T with covariates)",
                             "withoutComBat": "without ComBat"},
                         "ncol": 4,
                         "exclude_hue_value": []
                        }
    }
    
    #====================== 7: compare using classifier chain and adding the true GBM and IDH label ======================
    arrange_results_settings_dict["compare_CC_truelabel"]={
        "results_basepath": os.path.join(basepath, "7-compare_CC_truelabel"),
        "groupby_column": "base_task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "task_additional_description",
                         "rename_hue_values": {" withAge":"base", 
                                              " CC-withTrueLable": "With true tumor grade and true IDH labels"},
                         "ncol": 2,
                         "exclude_hue_value": []
                        }
    }


    #====================== 7: compare different feature selection method ======================
    arrange_results_settings_dict["compare_featureSelection"]={
        "results_basepath": os.path.join(basepath, "7_compare_featureSelect"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "feature_selection",
                         "rename_hue_values": {"AnovaTest":"Anova Test", 
                                              "MutualInformation": "Mutual Information",
                                              "PCA": "PCA",
                                              "SelectFromModel":"Select From Model",
                                              "RFE": "RFE"},
                         "ncol": 4,
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
                             " withAge":"Without tumor grade and IDH labels", 
                             " CC-withPredictLable": "with predicted GBM and IDH labels",
                             " CC-withTrueLable": "With true tumor grade and IDH labels"
                         },
                         "ncol": 3,
                         "exclude_hue_value": []
                        }
    }

    #====================== 9: compare results of different random seeds. ======================
    arrange_results_settings_dict["compare_random_seed"]={
        "results_basepath": os.path.join(basepath, "compare_RandomSeeds"),
        "groupby_column": "task",
        "plot_setting": {"x_column": "classifier", 
                         "hue_column": "additional_description",
                         "rename_hue_values": {
                             "randomseed0":"random seed 0", 
                             "randomseed500":"random seed 500", 
                             "randomseed2021":"random seed 2021", 
                             "randomseed5000":"random seed 5000", 
                             "randomseed10000":"random seed 10000", 
                         },
                         "ncol": 5,
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
    intensity_normalization="zscore"
    random_seed=0
    
    results_base_path= os.path.join(base_dataPath, "Results_randomseed"+str(random_seed)) 
    
    convert_binary_to_multiclass_setting_dict={}
    
    convert_binary_to_multiclass_setting_dict["TCGA-IDH"]={
        # Image filter dict
        "image_filter_dict": {"is_GBM": "original",
                             "is_IDH_mutant": "squareroot", 
                             "is_1p19q_codeleted": "log-sigma-1-0-mm-3D"},
        
        # The final setting name for each task
        "final_task_setting_dict":{"is_GBM": "TCGA_1.104.02_isGBM_withAge",
                                  "is_IDH_mutant": "TCGA_2.106_isIDHMutant_CC-withPredictLable_predictedForTrain", 
                                  "is_1p19q_codeleted": "TCGA_3.106_is1p19qCodeleted_CC-withPredictLable_predictedForTrain"},
        
        # base path to save the results.
        "save_results_basepath": results_base_path,                                    
    }
    
     ##=============== Add other distributions =============================
    for setting_name, convert_binary_to_multiclass_setting in convert_binary_to_multiclass_setting_dict.items():
        image_filter_dict=convert_binary_to_multiclass_setting["image_filter_dict"]
        final_task_setting_dict=convert_binary_to_multiclass_setting["final_task_setting_dict"]
        
        binary_task_path_dict={}
        ground_truth_target_excel_dict={}
        for task_name, image_filter in image_filter_dict.items():
            image_filter=image_filter_dict[task_name]
            final_task_setting=final_task_setting_dict[task_name]
            
            binary_task_path_dict[task_name]= os.path.join(results_base_path, image_filter, "TCGA_IDH-extracted_features-"+intensity_normalization, "withoutComBat-IgnoreDataImbalance", final_task_setting)
            
            # Feature folder name
            if image_filter.startswith("wavelet"):
                feature_folder="final_metadata(wavelet)"
            elif image_filter in ["original", "gradient", "log-sigma-1-0-mm-3D", "log-sigma-3-0-mm-3D"]:
                feature_folder="final_metadata(original)"
            elif image_filter in ["square", "squareroot", "logarithm", "exponential", "lbp-3D-m2", "lbp-3D-m1", "lbp-3D-k"]:
                feature_folder='final_metadata(exponential)'

        
            # excel path which saves the ground truth labels;
            ground_truth_target_excel_dict[task_name]={
                #train data
                "train_data": os.path.join(base_dataPath, "Features", feature_folder, intensity_normalization, "TCGA_extracted_features_IDH_train_resplited_randomseed_"+str(random_seed)+".xlsx"),
                #test data                            
                "test_data": os.path.join(base_dataPath, "Features", feature_folder, intensity_normalization, "TCGA_extracted_features_IDH_test_resplited_randomseed_"+str(random_seed)+".xlsx")
            }
         
        
        # folders of the tasks.
        convert_binary_to_multiclass_setting["binary_task_path_dict"]=binary_task_path_dict
        convert_binary_to_multiclass_setting["ground_truth_target_excel_dict"]=ground_truth_target_excel_dict["is_GBM"]
        convert_binary_to_multiclass_setting_dict[setting_name]=convert_binary_to_multiclass_setting
        
    return convert_binary_to_multiclass_setting_dict



#================ used for "main_plot_ROCs.ipynb" ======================
"""
Settings for plotting ROC curves with different random seeds in one plot;
"""
def get_plot_ROC_setting_dict():
    
    # base path
    base_dataPath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"

    plot_ROC_setting_dict={}
    
    plot_ROC_setting_dict["TCGA-IDH"]={
        "base_dataPath": base_dataPath,
        "random_seed_list": [0, 500, 2021, 5000, 10000],   
        "save_results_basepath": base_dataPath,     
        "data_excel_name": "multiclass_predicted_results-test_data.xlsx", #"multiclass_predicted_results-train_data.xlsx"
        "task_list": ["is_GBM", "is_IDH_mutant", "is_1p19q_codeleted"],
    }
    

    return plot_ROC_setting_dict

