#!/usr/bin/env python
# coding: utf-8

"""
Settings for encoding binary category data;
Encode the category data using label-encoder.
"""
def get_category_data_encoder_setting_dict():
    # base path
    base_dataPath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes/Features"
    category_data_encoder_setting_dict={}
    
    category_data_encoder_setting_dict["encode_category_data_TCGA-IDH"]={
        "original_data_excel": base_dataPath+"/gene_label/TCGA_subtypes.xlsx",
        "save_encoded_data_excel_path":base_dataPath+"/gene_label/TCGA_subtypes_IDH.xlsx",
        "category_column_list": ["Gender", "Study", "IDH.status",  "X1p.19q.codeletion"],
        "drop_nan_column_list": ["Gender", "Study", "IDH.status",  "X1p.19q.codeletion"],
        "column_rename_dict":{"Age..years.at.diagnosis.":"age",
                             "Gender_female":"is_female",
                             "Study_Glioblastoma multiforme":"is_GBM",
                             "IDH.status_Mutant":"is_IDH_mutant",
                             "X1p.19q.codeletion_codel":"is_1p19q_codeleted",
                        }
    }
    
    
    category_data_encoder_setting_dict["encode_category_data_TCGA-MGMT"]={
        "original_data_excel": base_dataPath+"/gene_label/TCGA_subtypes.xlsx",
        "save_encoded_data_excel_path":base_dataPath+"/gene_label/TCGA_subtypes_MGMT.xlsx",
        "category_column_list": ["Gender", "Study", "MGMT.promoter.status"],
        "drop_nan_column_list": ["Gender", "Study", "MGMT.promoter.status"],
        "column_rename_dict":{"Age..years.at.diagnosis.":"age",
                             "Gender_female":"is_female",
                             "Study_Glioblastoma multiforme":"is_GBM",
                             "MGMT.promoter.status_Methylated": "is_MGMT_Methylated"
                        }
    }
        
    return category_data_encoder_setting_dict

