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
Merge excel files by rows, from the LGG data and GBM data.
"""
def get_feature_merge_settings_dict1():
    
    basepath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    feature_merge_settings_dict={}
    
#     #=============== Merge the gene data for LGG and GBM data===================
#     feature_merge_settings_dict["merge_gene_data"]={
#         #----                                   
#         "excel_dict": {
#         "gbm_subtypes": basepath+"/originalData/TCGA/TCGA_GeneData/From_TCGAbiolinks/gbm_subtype.xlsx",
#         "lgg_subtypes": basepath+"/originalData/TCGA/TCGA_GeneData/From_TCGAbiolinks/lgg_subtype.xlsx"},
#         #----
#         "save_excel_path": basepath+"/Features/gene_label/TCGA_subtypes.xlsx",
#         "index_column_name": "patient_id",
#         "axis": 0,
#         "join": "outer"}
    
#     #=============== Merge the scanner info data for LGG and GBM data=================== 
#     feature_merge_settings_dict["merge_scanner_info_data"]={
#         #----                                   
#         "excel_dict": {
#         "gbm_subtypes": basepath+"/Features/scanner_info/Scanner_info_TCGA_GBM.xlsx",
#         "lgg_subtypes": basepath+"/Features/scanner_info/Scanner_info_TCGA_LGG.xlsx"},
#         #----
#         "save_excel_path": basepath+"/Features/scanner_info/Scanner_info.xlsx",
#         "index_column_name": "patient_id",
#         "axis": 0,
#         "join": "outer"}
    
    #=============== Merge the public features of the train data for LGG and GBM data=================== 
    feature_merge_settings_dict["merge_public_features_train"]={
        #----                                   
        "excel_dict": {
        "gbm_public_features": basepath+"/Features/public_features/TCGA_GBM_radiomicFeatures_train.xlsx",
        "lgg_public_features": basepath+"/Features/public_features/TCGA_LGG_radiomicFeatures_train.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/public_features/features_TCGA_train.xlsx",
        "index_column_name": "patient_id",
        "axis": 0,
        "join": "outer"}
    
    #=============== Merge the public features of the test data for LGG and GBM data=================== 
    feature_merge_settings_dict["merge_public_features_test"]={
        #----                                   
        "excel_dict": {
        "gbm_public_features": basepath+"/Features/public_features/TCGA_GBM_radiomicFeatures_test.xlsx",
        "lgg_public_features":basepath+"/Features/public_features/TCGA_LGG_radiomicFeatures_test.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/public_features/features_TCGA_test.xlsx",
        "index_column_name": "patient_id",
        "axis": 0,
        "join": "outer"}
     
    return feature_merge_settings_dict
    
    

"""
Merge excel files for the TCGA data.
"""
def get_feature_merge_settings_dict():
    basepath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    feature_merge_settings_dict={}
    
    #1) With features extracted from the .nii images by ourselves, for predicting IDH status.
    feature_merge_settings_dict["TCGA_extracted_features_IDH_train"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/extracted_features/features_TCGA_train.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_IDH.xlsx"}}
    
    feature_merge_settings_dict["TCGA_extracted_features_IDH_test"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/extracted_features/features_TCGA_test.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_IDH.xlsx"}}
    
    #2) With features extracted from the .nii images by ourselves, for predicting MGMT status.
    feature_merge_settings_dict["TCGA_extracted_features_MGMT_train"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/extracted_features/features_TCGA_train.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_MGMT.xlsx"}}
    
    feature_merge_settings_dict["TCGA_extracted_features_MGMT_test"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/extracted_features/features_TCGA_test.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_MGMT.xlsx"}}
    
   #3) With features extracted from the .nii images by ourselves, for predicting IDH status.
    feature_merge_settings_dict["TCGA_public_features_IDH_train"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/public_features/features_TCGA_train.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_IDH.xlsx"}}
    
    feature_merge_settings_dict["TCGA_public_features_IDH_test"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/public_features/features_TCGA_test.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_IDH.xlsx"}}
    
    #4) With features extracted from the .nii images by ourselves, for predicting MGMT status.
    feature_merge_settings_dict["TCGA_public_features_MGMT_train"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/public_features/features_TCGA_train.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_MGMT.xlsx"}}
    
    feature_merge_settings_dict["TCGA_public_features_MGMT_test"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/public_features/features_TCGA_test.xlsx",
        "scanner_info": basepath+"/Features/scanner_info/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes_MGMT.xlsx"}}
    
    
     ##=============== Add other distributions =============================
    for setting_name, feature_merge_settings in feature_merge_settings_dict.items():
        feature_merge_settings["save_excel_path"]=basepath+"/Features/final_metadata/"+setting_name+".xlsx"
        feature_merge_settings["index_column_name"]="patient_id"
        feature_merge_settings["axis"]=1
        feature_merge_settings["join"]="inner"
        feature_merge_settings_dict[setting_name]=feature_merge_settings
        
    return feature_merge_settings_dict


    
