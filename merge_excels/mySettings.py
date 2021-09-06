#!/usr/bin/env python
# coding: utf-8

"""
Merge features for the MGMT prediction task of BraTS2021 competition.
"""
def get_feature_merge_settings_dict1():
    
    basepath="G://PhDProjects/RadiogenomicsProjects/BraTS2021/Features"
    feature_merge_settings_dict={}
    
    #============= Extract the features directly from the .nii data from BraTS2021 segmentation task. ==============
    feature_merge_settings_dict["BraTS2021_MGMT_train"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/extracted_features/features_BraTS2021_train.xlsx",
        "scanner_info": basepath+"/scanner_info/scanner_info_BraTS2021_train.xlsx",
        "MGMT_label": basepath+"/MGMT_info/MGMT_info_train.xlsx"},
        #----
        "save_excel_path": basepath+"/final_metadata/features_BraTS2021_train.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
    
    feature_merge_settings_dict["BraTS2021_MGMT_validation"]={
        #----
        "excel_dict": {
        "feature_data": basepath+"/extracted_features/features_BraTS2021_validation.xlsx",
        "scanner_info": basepath+"/scanner_info/scanner_info_BraTS2021_validation.xlsx",
        "MGMT_label": None},
        #----
        "save_excel_path": basepath+"/final_metadata/features_BraTS2021_validation.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
    
    
    #======= Convert the .dcm images to .nii images of BraTS2021 MGMT competition, ========
    #=======   and then extract the features from the converted .nii images =====
    feature_merge_settings_dict["BraTS2021_MGMT_train_dcm_to_nii"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/extracted_features/features_BraTS2021_train_dcm_to_nii.xlsx",
        "scanner_info": basepath+"/scanner_info/scanner_info_BraTS2021_train.xlsx",
        "MGMT_label": basepath+"/MGMT_info/MGMT_info_train.xlsx"},
        #----
        "save_excel_path": basepath+"/final_metadata/features_BraTS2021_train_dcm_to_nii.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
    
    feature_merge_settings_dict["BraTS2021_MGMT_validation_dcm_to_nii"]={
        #----
        "excel_dict": {
        "feature_data": basepath+"/extracted_features/features_BraTS2021_validation_dcm_to_nii.xlsx",
        "scanner_info": basepath+"/scanner_info/scanner_info_BraTS2021_validation.xlsx",
        "MGMT_label": None},
        #----
        "save_excel_path": basepath+"/final_metadata/features_BraTS2021_validation_dcm_to_nii.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
    
    return feature_merge_settings_dict


"""
Merge excel files for the TCGA data, with features extracted from the .nii images by ourselves.
"""
def get_feature_merge_settings_dict():
    
    basepath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    feature_merge_settings_dict={}
    
    feature_merge_settings_dict["merge_gene_data"]={
        #----                                   
        "excel_dict": {
        "gbm_subtypes": basepath+"/originalData/TCGA/TCGA_GeneData/From_TCGAbiolinks/gbm_subtype.xlsx",
        "lgg_subtypes": basepath+"/originalData/TCGA/TCGA_GeneData/From_TCGAbiolinks/lgg_subtype.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/gene_label/TCGA_subtypes.xlsx",
        "index_column_name": "patient_id",
        "axis": 0,
        "join": "outer"}
    
    
    feature_merge_settings_dict["merge_scanner_info_data"]={
        #----                                   
        "excel_dict": {
        "gbm_subtypes": basepath+"/Features/scanner_info/Scanner_info_TCGA_GBM.xlsx",
        "lgg_subtypes": basepath+"/Features/scanner_info/Scanner_info_TCGA_LGG.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "index_column_name": "patient_id",
        "axis": 0,
        "join": "outer"}
        
        
    feature_merge_settings_dict["TCGA_train"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/extracted_features/features_TCGA_train.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_encoded.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/final_metadata/features_TCGA_train.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
    
    feature_merge_settings_dict["TCGA_test"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/extracted_features/features_TCGA_test.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_encoded.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/final_metadata/features_TCGA_test.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
   
    
    return feature_merge_settings_dict
    
    




    
