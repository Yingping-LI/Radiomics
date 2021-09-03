#!/usr/bin/env python
# coding: utf-8

"""
Merge features for the MGMT prediction task of BraTS2021 competition.
"""
def get_feature_merge_settings_dict1():
    
    basepath="G://DURING PHD/5)Glioblastoma_MGMT_RSNA-MICCAI/Features"
    feature_merge_settings_dict={}
    
    feature_merge_settings_dict["BraTS2021_MGMT_train"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/extracted_features/features_BraTS2021_MGMT_train.xlsx",
        "scanner_info": basepath+"/scanner_info/scanner_info_train.xlsx",
        "MGMT_label": basepath+"/MGMT_info/MGMT_info_train.xlsx"},
        #----
        "save_excel_path": basepath+"/final_metadata/features_BraTS2021_MGMT_train.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
    
    feature_merge_settings_dict["BraTS2021_MGMT_validation"]={
        #----
        "excel_dict": {
        "feature_data": basepath+"/extracted_features/features_BraTS2021_MGMT_validation.xlsx",
        "scanner_info": basepath+"/scanner_info/scanner_info_validation.xlsx",
        "MGMT_label": None},
        #----
        "save_excel_path": basepath+"/final_metadata/features_BraTS2021_MGMT_validation.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
    
    return feature_merge_settings_dict


"""
Merge excel files for the TCGA data.
"""
def get_feature_merge_settings_dict():
    
    basepath="C://YingpingLI/Glioma/TCGA"
    feature_merge_settings_dict={}
    
    feature_merge_settings_dict["merge_gene_data"]={
        #----                                   
        "excel_dict": {
        "gbm_subtypes": basepath+"/TCGA_GeneData/From_TCGAbiolinks/gbm_subtype.xlsx",
        "lgg_subtypes": basepath+"/TCGA_GeneData/From_TCGAbiolinks/lgg_subtype.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/gene_label/TCGA_subtypes.xlsx",
        "index_column_name": "patient_id",
        "axis": 0,
        "join": "outer"}
    
    feature_merge_settings_dict["TCGA_train"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/extracted_features/features_TCGA_train.xlsx",
        "scanner_info": basepath+"/TCIA_Segmentation/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/final_metadata/features_TCGA_train.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
    
    feature_merge_settings_dict["TCGA_test"]={
        #----                                   
        "excel_dict": {
        "feature_data": basepath+"/Features/extracted_features/features_TCGA_test.xlsx",
        "scanner_info": basepath+"/TCIA_Segmentation/Scanner_info.xlsx",
        "gene_label": basepath+"/Features/gene_label/TCGA_subtypes.xlsx"},
        #----
        "save_excel_path": basepath+"/Features/final_metadata/features_TCGA_test.xlsx" ,
        "index_column_name": "patient_id",
        "axis": 1,
        "join": "inner"}
   
    
    return feature_merge_settings_dict
    
    




    
