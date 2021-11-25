#!/usr/bin/env python
# coding: utf-8

import os

#============================   Used for main_visualize_images_and_tumors.ipynb =================================
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






#============================ used for main_visualize_variable_ratios.ipynb =================================
def get_variable_ratio_visualization_setting_dict():
    # base path
    base_dataPath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    
    #define settings for visualizing classification target ratios.
    variable_ratio_visualization_setting_dict={}
    
   #=================== For predicting IDH mutation status ==========================
    variable_ratio_visualization_setting_dict["TCGA_IDH"]={
        "data_excel_path": base_dataPath+"/Features/gene_label/TCGA_subtypes_IDH.xlsx",
        #settings for cacluating cross table.
        "crosstab_setting_dict": {
            # Cross table for statistics
            "overview":
            {"index": ["Tumor Grade"],
             "columns": ["IDH.status",  "X1p.19q.codeletion"],
             "stacked_for_plots":False},
            # Plot for tumor grade
            "tumor_grade":
            {"index": ["Tumor Grade"],
             "columns": ["Tumor Grade"],
             "stacked_for_plots":True}, 
            # plot to show tumor_grade vs. IDH
            "tumor_grade_vs_IDH":
            {"index": ["Tumor Grade"],
             "columns": ["IDH.status"],
             "stacked_for_plots":False},
            # plot to show tumor_grade vs. IDH
            "IDH_vs_1p19q":
            {"index": ["IDH.status"],
             "columns": ["X1p.19q.codeletion"],
             "stacked_for_plots":False},
            # Plot for IDH mutation status
            "IDH":
            {"index": ["IDH.status"],
             "columns": ["Tumor Grade"],
             "stacked_for_plots":True},           
            # Plot for 1p/19q mutation status
            "1p19q":
            {"index": ["X1p.19q.codeletion"],
             "columns": ["Tumor Grade", "IDH.status"],
             "stacked_for_plots":True},
        },
        #settings for ploting the category data.
        "visualize_category_setting_list":[
            #{"x": "Tumor Grade", "hue": None},
            #{"x": "Tumor Grade", "hue": "IDH.status"},
            ],
    
        "save_basepath": base_dataPath+"/Features/gene_label"
    }
       
    #=================== For predicting MGMT methylation status ==========================
    variable_ratio_visualization_setting_dict["TCGA_MGMT"]={
        "data_excel_path": base_dataPath+"/Features/gene_label/TCGA_subtypes_MGMT.xlsx",
        #settings for cacluating cross table.
        "crosstab_setting_dict": {
            # Cross table for statistics
            "tumor_grade":
            {"index": ["Tumor Grade"],
             "columns": ["MGMT.promoter.status"],
             "stacked_for_plots":False},
            # Plot for MGMT mutation status
            "MGMT":
            {"index": ["MGMT.promoter.status"],
             "columns": ["Tumor Grade"],
             "stacked_for_plots":True}
        },
        #settings for ploting the category data.
        "visualize_category_setting_list":[
            #{"x": "MGMT.promoter.status", "hue": None},
            #{"x": "Tumor Grade", "hue": "MGMT.promoter.status"}
        ],
        "save_basepath": base_dataPath+"/Features/gene_label"
    }

    return variable_ratio_visualization_setting_dict