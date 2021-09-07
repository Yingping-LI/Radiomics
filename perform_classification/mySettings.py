#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os

'''
Basic Settings for the code
'''
global_basic_settings={
    "experiment_class": "BraTS2021", #"BraTS2021",  "TCGA_IDH",  "TCGA_MGMT"
    ##"501.01_BraTS2021-segNiiData_base", "601.01_BraTS2021-dcmToNiiData_base",  "701.01_BraTS2021-segNiiData-zscore_base"
    "task_name":"BraTS2021_501.01_segNiiData_base", 
    "feature_selection_method":"AnovaTest", #"RFECV","RFE", AnovaTest, SelectFromModel
    "use_randomSearchCV":True, #False, True
    "harmonization_method": "withoutComBat", # withoutComBat, "parametric_ComBat", nonParametric_ComBat, noEB_ComBat
    "harmonization_label": "is_3T",      #"Tissue.source.site", "is_3T"
}


def get_basic_settings():
   
    return global_basic_settings

def get_classification_task_settings():
    basic_settings=get_basic_settings()
    experiment_class=basic_settings["experiment_class"]
    task_name=basic_settings["task_name"]
    
    if experiment_class=="BraTS2021":
        classification_tasks_dict=get_classification_tasks_dict_BraTS2021()
        
    elif experiment_class=="TCGA_IDH":
        classification_tasks_dict=get_classification_tasks_dict_TCGA_IDH()  
        
    elif experiment_class=="TCGA_MGMT":
        classification_tasks_dict=get_classification_tasks_dict_TCGA_MGMT()  
    
    classification_task_settings=classification_tasks_dict[task_name]
    
    return task_name, classification_task_settings

#=================================================   Data preprocessing  ==================================================
def convert_complex_to_real(datadf, feature_namelist):
    # convert_complex_to_real
    datadf_converted=datadf.copy()
    datadf_converted[feature_namelist]=datadf[feature_namelist].applymap(lambda x: complex(x).real)
    
    return datadf_converted

def preprocessing_data(datadf, feature_columns):
     #convert the complex data to real data
    datadf=convert_complex_to_real(datadf, feature_columns)
    
    #fill nan values with 0.
    datadf.fillna(value=0, inplace=True)
    
    return datadf
#======================================= BraTS2021 classification task settings =======================================================
"""
Perform MGMT classification for BraTS2021 competition.
"""
def get_classification_tasks_dict_BraTS2021():
    basepath="G://PhDProjects/RadiogenomicsProjects/BraTS2021"
    basic_settings=get_basic_settings() 
    base_results_path=basepath+"/Results/Results_BraTS2021_MGMT/"+basic_settings["harmonization_method"]+"_"+basic_settings["harmonization_label"]
    classification_tasks_dict={}
    
    
    ## basic excel path settings
    classification_tasks_dict["BraTS2021_501.01_segNiiData_base"]={
        "train_excel_path": basepath+"/Features/final_metadata/features_BraTS2021_train.xlsx",
        "test_excel_path_dict": {"test_data": basepath+"/Features/final_metadata/features_BraTS2021_validation.xlsx"},
    }
    
    classification_tasks_dict["BraTS2021_601.01_dcmToNiiData_base"]={
        "train_excel_path": basepath+"/Features/final_metadata/features_BraTS2021_train_dcm_to_nii.xlsx",
        "test_excel_path_dict": {"test_data": basepath+"/Features/final_metadata/features_BraTS2021_validation_dcm_to_nii.xlsx"},
    }
    
#     classification_tasks_dict["BraTS2021_701.01_segNiiData-zscore_base"]={
#         "train_excel_path": basepath+"/Features/final_metadata/features_BraTS2021_train_zscore.xlsx",
#         "test_excel_path_dict": {"test_data": basepath+"/Features/final_metadata/features_BraTS2021_validation_zscore.xlsx"},
#     }
           
    
    ## Other settings like "train_data", "test_data_dict", "label_column", "base_results_path", "feature_columns"
    for task_name, classification_settings in classification_tasks_dict.items():
        train_excel_path=classification_settings["train_excel_path"]
        test_excel_path_dict=classification_settings["test_excel_path_dict"]

        # preprocessing train data
        train_data=pd.read_excel(train_excel_path, index_col=0)
        feature_columns=get_feature_columns(train_data)
        train_data=preprocessing_data(train_data, feature_columns)
        
        # preprocessing test data
        test_data_dict={}
        for description, test_excel_path in test_excel_path_dict.items():
            test_data=pd.read_excel(test_excel_path, index_col=0)
            test_data=preprocessing_data(test_data, feature_columns)
            test_data_dict[description]=test_data
        
        # set and save the settings
        classification_settings["train_data"]=train_data
        classification_settings["test_data_dict"]=test_data_dict
        classification_settings["label_column"]="MGMT_value" 
        classification_settings["base_results_path"]=base_results_path
        classification_settings["feature_columns"]=feature_columns
        classification_tasks_dict[task_name]=classification_settings
        
        if not os.path.exists(base_results_path):
            os.makedirs(base_results_path)
    
    return classification_tasks_dict


def get_feature_columns(train_data, modality_list=["t1", "t1ce", "t2", "flair"], tumor_subregion_list=None):
    all_columns=train_data.columns
    
    # 1. filter features from different modalities;
    modalities_feature_columns=[]
    for column in all_columns:
        column_name_split=column.split("_")
        for modality_name in modality_list:
            if column.startswith(modality_name+"_"): 
                modalities_feature_columns.append(column)
                
    # 2. filter features from different tumor subregions;
    if tumor_subregion_list is None:
        subregion_feature_columns=modalities_feature_columns
    else:
        subregion_feature_columns=[]
        for column in modalities_feature_columns:
            column_name_split=column.split("_")
            if len(column_name_split)>1 and column_name_split[1] in tumor_subregion_list:
                subregion_feature_columns.append(column)
      
    #3. the shape feature are the same for all modalities, so only keep them for one modality
    shape_features=[]
    for column in subregion_feature_columns:
        column_name_split=column.split("_")
        for modality_name in modality_list[:-1]:
            if column.startswith(modality_name+"_") and column_name_split[3]=="shape":
                shape_features.append(column)
     
    final_feature_columns=list(set(subregion_feature_columns).difference(set(shape_features)))                
    print("There are {} radiomic features in total!".format(len(final_feature_columns)))
    
    return final_feature_columns
    
    
    
    
#=====================  classification task settings for TCGA datast: GBM, IDH and 1p/19q ========================================
def get_classification_tasks_dict_TCGA_IDH():
    # basic paths
    basepath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes" 
    basic_settings=get_basic_settings() 
    base_results_path=basepath+"/Results/TCGA_subtypes/"+basic_settings["harmonization_method"]+"_"+basic_settings["harmonization_label"]
    
    ##--------------------- Prepare the data -------------------
    # data path
    data_excel_path=basepath+"/TCGA/TCGA_IDH_well_arranged_data_withScannerInfo.xlsx"

    #read the preprocessing the data
    dataframe=pd.read_excel(data_excel_path, index_col=0)
    feature_namelist, ET_related_feature_namelist, clinical_namelist, classification_label_namelist=get_column_list_for_TCGA(dataframe)
       
    ## split the train and test data accoring to the public split of TCGA-GBM and TCGA-LGG dataset.
    train_data=dataframe[dataframe["train_test_class"] == "train"]
    test_data=dataframe[dataframe["train_test_class"] == "test"]
    print("\n****Train and test set split!**** \nTrain: {} patients; \nTest: {} patients.".format(train_data.shape[0], test_data.shape[0]))
 
    ## Filter the LGG data for predicting 1p/19q status.
    train_data_LGG=train_data[train_data["is_GBM"] == 0]
    test_data_LGG=test_data[test_data["is_GBM"] == 0]
    print("\n****Train and test set split for LGG patients!**** \nTrain: {} patients; \nTest: {} patients.".format(train_data_LGG.shape[0], test_data_LGG.shape[0]))

    ## Filter the GBM data for predicting MGMT methylation status.
    train_data_GBM=train_data[train_data["is_GBM"] == 1]
    test_data_GBM=test_data[test_data["is_GBM"] == 1]
    print("\n****Train and test set split for GBM patients!**** \nTrain: {} patients; \nTest: {} patients.".format(train_data_GBM.shape[0], test_data_GBM.shape[0]))

    #-------------------------- Define the classification tasks ----------------------------------
    classification_tasks_dict={}
    #========= predict LGG vs. GBM  ========
    classification_tasks_dict["TCGA_1.01_isGBM_base"]={
        "train_data": train_data, 
        "test_data": test_data, 
        "feature_columns":feature_namelist, 
        "label_column":"is_GBM",
        "base_results_path":base_results_path}
    
    classification_tasks_dict["TCGA_1.02_isGBM_with_clinicalInfo"]={
        "train_data": train_data, 
        "test_data": test_data, 
        "feature_columns":feature_namelist+clinical_namelist,
        "label_column":"is_GBM",
        "base_results_path":base_results_path}
    

    #========= predict IDH mutation status  ========= 
    classification_tasks_dict["TCGA_2.01_isIDHMutant_base"]= {
        "train_data": train_data, 
        "test_data": test_data, 
        "feature_columns":feature_namelist, 
        "label_column":"is_IDH_mutant",
        "base_results_path":base_results_path}

    classification_tasks_dict["TCGA_2.02_isIDHMutant_with_clinicalInfo"]={
        "train_data": train_data, 
        "test_data": test_data, 
        "feature_columns":feature_namelist+clinical_namelist,
        "label_column":"is_IDH_mutant",
        "base_results_path":base_results_path}
        
    #========= predict 1p/19q codeletion status ========= 
    classification_tasks_dict["TCGA_3.01_is1p19qCodeleted_base"]={
        "train_data": train_data_LGG, 
        "test_data": test_data_LGG, 
        "feature_columns":feature_namelist, 
        "label_column":"is_1p19q_codeleted",
        "base_results_path":base_results_path}
        
    classification_tasks_dict["TCGA_3.02_is1p19qCodeleted_with_clinicalInfo"]={
        "train_data": train_data_LGG, 
        "test_data": test_data_LGG, 
        "feature_columns":feature_namelist+clinical_namelist, 
        "label_column":"is_1p19q_codeleted",
        "base_results_path":base_results_path}
    
    return classification_tasks_dict
                              
                              
#=====================  classification task settings for TCGA datast: MGMT ========================================
def get_classification_tasks_dict_TCGA_MGMT():
    # basic paths
    basepath="G://PhDProjects/RadiogenomicsProjects/GliomasSubtypes"
    basic_settings=get_basic_settings() 
    base_results_path=basepath+"/Results/TCGA_MGMT/"+basic_settings["harmonization_method"]+"_"+basic_settings["harmonization_label"]
    
    ##--------------------- Prepare the data -------------------
    # data path
    data_excel_path=basepath+"/TCGA/TCGA_MGMT_well_arranged_data_withScannerInfo.xlsx",

    #read the preprocessing the data
    dataframe=pd.read_excel(data_excel_path, index_col=0)
    feature_namelist, ET_related_feature_namelist, clinical_namelist, classification_label_namelist=get_column_list_for_TCGA(dataframe)
       
    ## split the train and test data accoring to the public split of TCGA-GBM and TCGA-LGG dataset.
    train_data=dataframe[dataframe["train_test_class"] == "train"]
    test_data=dataframe[dataframe["train_test_class"] == "test"]
    print("\n****Train and test set split!**** \nTrain: {} patients; \nTest: {} patients.".format(train_data.shape[0], test_data.shape[0]))
 
    ## Filter the LGG data for predicting 1p/19q status.
    train_data_LGG=train_data[train_data["is_GBM"] == 0]
    test_data_LGG=test_data[test_data["is_GBM"] == 0]
    print("\n****Train and test set split for LGG patients!**** \nTrain: {} patients; \nTest: {} patients.".format(train_data_LGG.shape[0], test_data_LGG.shape[0]))

    ## Filter the GBM data for predicting MGMT methylation status.
    train_data_GBM=train_data[train_data["is_GBM"] == 1]
    test_data_GBM=test_data[test_data["is_GBM"] == 1]
    print("\n****Train and test set split for GBM patients!**** \nTrain: {} patients; \nTest: {} patients.".format(train_data_GBM.shape[0], test_data_GBM.shape[0]))

    #-------------------------- Define the classification tasks ----------------------------------
    classification_tasks_dict={}

    #========= predict MGMT methylated vs. unmethylated for LGG and GBM data ========
    classification_tasks_dict["TCGA_4.01_isMGMTMethylated_base"]: {
        "train_data": train_data, 
        "test_data": test_data, 
        "feature_columns":feature_namelist, 
        "label_column":"is_MGMT_Methylated",
        "base_results_path":base_results_path}

    classification_tasks_dict["TCGA_4.02_isMGMTMethylated_with_clinicalInfo"]={
        "train_data": train_data, 
        "test_data": test_data, 
        "feature_columns":feature_namelist+clinical_namelist,
        "label_column":"is_MGMT_Methylated",
        "base_results_path":base_results_path}

    #========= predict MGMT methylated vs. unmethylated for GBM data ========
    classification_tasks_dict["TCGA_5.01_GBM-isMGMTMethylated_base"]={
        "train_data": train_data_GBM, 
        "test_data": test_data_GBM, 
        "feature_columns":feature_namelist, 
        "label_column":"is_MGMT_Methylated",
        "base_results_path":base_results_path}

    classification_tasks_dict["TCGA_5.02_GBM-isMGMTMethylated_with_clinicalInfo"]={
        "train_data": train_data_GBM, 
        "test_data": test_data_GBM, 
        "feature_columns":feature_namelist+clinical_namelist,
        "label_column":"is_MGMT_Methylated",
        "base_results_path":base_results_path}

    #========= predict MGMT methylated vs. unmethylated for GBM data, with ET features ========
    classification_tasks_dict["TCGA_6.01_GBM-isMGMTMethylated-withETFeatures_base"]={
        "train_data": train_data_GBM, 
        "test_data": test_data_GBM, 
        "feature_columns":feature_namelist+ET_related_feature_namelist, 
        "label_column":"is_MGMT_Methylated",
        "base_results_path":base_results_path}

    classification_tasks_dict["TCGA_6.02_GBM-isMGMTMethylated-withETFeatures_with_clinicalInfo"]= {
        "train_data": train_data_GBM, 
        "test_data": test_data_GBM, 
        "feature_columns":feature_namelist+ET_related_feature_namelist+clinical_namelist, 
        "label_column":"is_MGMT_Methylated",
        "base_results_path":base_results_path}
    
    return classification_tasks_dict

  
"""
For TCGA dataset, classify the columns into features, interested clinical info and the classification targets.
"""
def get_column_list_for_TCGA(columns):

    feature_namelist=[]
    clinical_namelist=[]
    classification_label_namelist=[]
    ET_related_feature_namelist=[]
    for column in columns:
        if column.startswith("ET_related_feature_"): 
            ET_related_feature_namelist.append(column)
            
        elif column in ["age", "is_female"]:
            clinical_namelist.append(column)

        elif column in ["is_GBM", "is_IDH_mutant", "is_1p19q_codeleted", "is_MGMT_Methylated"]:
            classification_label_namelist.append(column)

        else:
            column_prefix=column.split("_")[0]
            if column_prefix in ["VOLUME", "DIST", "INTENSITY", "HISTO","SPATIAL", "ECCENTRICITY", "SOLIDITY", "TEXTURE", "TGM"]:
                feature_namelist.append(column)
 
    return feature_namelist, ET_related_feature_namelist, clinical_namelist, classification_label_namelist  
    
    