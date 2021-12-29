#!/usr/bin/env python
# coding: utf-8

# ### Perform feature selection and classification to predict the gene status of the gliomas patients.
# 
# For example, we can use this code and the radiomic features to predict:
# - 1. LGG vs. GBM;
# - 2. IDH mutant vs. IDH wildtype;
# - 3. 1p/19q codeleted vs. 1p/19q intact;
# - 4. MGMT methylated vs. MGMT unmethylated.


#=======================================
import os
import pandas as pd
import numpy as np
from time import time
import warnings


## For preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# For feature selection and classifier models
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFECV, RFE, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
#from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import feature_selection




from imblearn.pipeline import Pipeline 

## feature selection
from probatus.feature_elimination import ShapRFECV

## data imputation
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion

#import function from the self-defined utils
import sys
sys.path.append("../")
from utils.myUtils import mkdir, save_dict, load_dict, get_logger, save_pickle, load_pickle, save_log, traversalDir_FirstDir
from utils.harmonizationUtils import neuroComBat_harmonization, neuroComBat_harmonization_FromTraning
from myTransformers import ComBatTransformer, PandasSimpleImputer, SelectColumnsTransformer, DeleteCorrColumnTransformer
from myClassificationUtils import *
from mySettings import get_basic_settings

# ================= Global variable ==============
global random_seed
random_seed=get_basic_settings()["random_seed"]

# ============ Step 1: find the best model.  ==========================
# 
# - from a list of models, with different feature selection methods and classifiers;
# - using the train data and 5-folds cross-validation.

def hyperparameter_tuning_for_different_models(train_data, feature_columns, keep_feature_columns, label_column, harmonization_settings, save_results_path, feature_selection_type, imbalanced_data_strategy, searchCV_method="randomSearchCV"):
    """
    Tuning the hypperparameters for different models.
    """
    ## harmonization settings and data for cross validation.
    X=train_data
    y=train_data[label_column]
    harmonization_method=harmonization_settings["harmonization_method"]
    harmonization_label=harmonization_settings["harmonization_label"]
    harmonization_ref_batch=harmonization_settings["harmonization_ref_batch"]
    
    ##====== classifiers and parameters =======
    classification_models, param_grids=get_classifiers(random_seed)
    
    ## feature numbers for random search
    feature_number_for_selection=[20, 40, 60, 80, 100]
    
    ## scoring metric
    scoring="roc_auc"
    
    ## ============ Models =================
    for classfier_name, classifier_model in classification_models.items():
        save_log("\n\n ======Exploring the hyperparameters for feature_selection_method={}, classifier={}. =========".format(feature_selection_type, classfier_name))
        start_time = time()
        
            
        ###-----List of preprocessing transformers-------
        # Imputation
        #imputation_transformer =[('imputation', PandasSimpleImputer(strategy='constant', fill_value=0))]
        #imputation_transformer = FeatureUnion(transformer_list=[('features', SimpleImputer(strategy='mean')),
        #                        ('indicators', MissingIndicator(features="missing-only"))])
        imputation_transformer=[]
        
        # ComBat harmonization
        if harmonization_method!="withoutComBat":
            ComBat_transformer=ComBatTransformer(feature_columns, harmonization_label, harmonization_method, harmonization_ref_batch)
            harmonization_transformer=[("harmonization", ComBat_transformer)]
        else:
            harmonization_transformer=[]
            
        # Imbalanced data handler
        imbalanced_data_handler = get_imbalanced_data_handler(y, imbalanced_data_strategy, random_seed)
        
        # Scaler
        scaler_transformer=[('scaler', RobustScaler())]  # {MinMaxScaler(feature_range=(0,1)), RobustScaler(), StandardScaler()}
        
        # delete the highly correlated features.
        #delete_corr_features_transformer=[("del_corr_features", DeleteCorrColumnTransformer(threshold=0.95))]
        delete_corr_features_transformer= []
        
        ##-----
        preprocessing_transformer_list=imputation_transformer+harmonization_transformer+[
            ("filter_features", SelectColumnsTransformer(feature_columns))]+scaler_transformer+delete_corr_features_transformer
        
        cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        
        #--------------------- begin hyperparameter tuning process--------------------------------
        ### define feature selection function.
        if feature_selection_type=="RFE":
            feature_selection_method=RFE(estimator=classifier_model, step=5) #, n_features_to_select=20
            pipeline = Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            #save_log("Possible hyperparameters for {} pipeline: \n {}".format(classfier_name, pipeline.get_params().keys()))
            
            randomsearch_param_grids=dict(**{"feature_selection__n_features_to_select": feature_number_for_selection}, **{"feature_selection__estimator__"+key: item for key, item in param_grids[classfier_name].items()}) 
            
            search = searchCV(pipeline, randomsearch_param_grids, scoring, cross_val, random_seed).fit(X, y)
            n_feature_selected=search.best_estimator_["feature_selection"].n_features_
            
        elif feature_selection_type=="RFECV": 
            feature_selection_method=RFECV(estimator=classifier_model, step=5, min_features_to_select=20) 
            pipeline = Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            
            randomsearch_param_grids={"feature_selection__estimator__"+key: item for key, item in param_grids[classfier_name].items()}

            search = searchCV(pipeline, randomsearch_param_grids, scoring, cross_val, random_seed).fit(X, y)
            n_feature_selected=search.best_estimator_["feature_selection"].n_features_
            
        elif feature_selection_type=="SelectFromModel": 
            feature_selection_method=SelectFromModel(estimator=classifier_model) #max_features=20
            pipeline = Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            
           
            randomsearch_param_grids=dict(**{"feature_selection__max_features": feature_number_for_selection}, **{"feature_selection__estimator__"+key: item for key, item in param_grids[classfier_name].items()}) 
            
            search = searchCV(pipeline, randomsearch_param_grids, scoring, cross_val, random_seed).fit(X, y)
            n_feature_selected= search.best_estimator_.transform(X).shape[1]
            
        elif feature_selection_type=="AnovaTest" or feature_selection_type=="ChiSquare" or feature_selection_type=="MutualInformation": 
            
            if feature_selection_type=="AnovaTest":
                feature_selection_method=SelectKBest(score_func=f_classif) # k=n_features_to_select

            elif feature_selection_type=="ChiSquare":
                feature_selection_method=SelectKBest(score_func=chi2)

            elif feature_selection_type=="MutualInformation":
                feature_selection_method=SelectKBest(score_func=mutual_info_classif) 
            
            # Pipeline
            selected_features=Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            combined_features = FeatureUnion([("keep_features_directly", SelectColumnsTransformer(keep_feature_columns)), ("selected_features", selected_features)])
            pipeline = Pipeline(steps=[('features',combined_features)]+imbalanced_data_handler+[('classifier',classifier_model)])           
            
            # random search parameters
            randomsearch_param_grids=dict(**{"features__selected_features__feature_selection__k": feature_number_for_selection}, **{"classifier__"+key: item for key, item in param_grids[classfier_name].items()}) 
            
            # grid search
            search =searchCV(pipeline, randomsearch_param_grids, scoring, cross_val, random_seed, searchCV_method).fit(X, y)
            n_feature_selected=search.best_estimator_["features"].get_params()["selected_features"]["feature_selection"].k
            
        elif feature_selection_type=="PCA":
            feature_selection_method=PCA()
            
            # Pipeline
            selected_features=Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            combined_features = FeatureUnion([("keep_features_directly", SelectColumnsTransformer(keep_feature_columns)), ("selected_features", selected_features)])
            pipeline = Pipeline(steps=[('features',combined_features), ('classifier',classifier_model)]) 
            
            # random search parameters
            randomsearch_param_grids=dict(**{"features__selected_features__feature_selection__n_components": feature_number_for_selection}, **{"classifier__"+key: item for key, item in param_grids[classfier_name].items()})

            search = searchCV(pipeline, randomsearch_param_grids, scoring, cross_val, random_seed).fit(X, y)
            n_feature_selected=search.best_estimator_["features"].get_params()["selected_features"]["feature_selection"].n_components

        else:
            raise Exception("Undefined feature selection function: {} !!".format(feature_selection_type))
        
        
        #--------------------- end hyperparameter tuning process--------------------------------
        ##plot the grid search results
        if searchCV_method=="gridSearchCV":
            save_hyperparameter_tuning_basepath=os.path.join(save_results_path, "hyperparameter_tuning")
            mkdir(save_hyperparameter_tuning_basepath)
            plot_GridSearch_results(search, os.path.join(save_hyperparameter_tuning_basepath, "hyperparam_tuning_"+classfier_name+".jpeg"))
        
        ### get the best estimator.
        if feature_selection_type=="SelectFromModel":
            best_estimator=Pipeline(steps=preprocessing_transformer_list
                                    +[('feature_selection',search.best_estimator_['feature_selection']),
                                   ('classifier',search.best_estimator_['feature_selection'].estimator_)])
        else:
            best_estimator= search.best_estimator_
         
        
        ### arrange the results and save it into a dict.
        save_classifier_name=feature_selection_type+"_"+classfier_name
        result={'classfier_name':save_classifier_name,
                'best score': search.best_score_, 
                'best params': search.best_params_,
                'time_cost':time()-start_time,
                'n_feature_selected': n_feature_selected,
                #'grid': search, 
                'best_estimator': best_estimator,
                #'cv': search.cv,
                'scorer':search.scorer_,
                'cv_results_': pd.DataFrame(search.cv_results_) 
                }
 
        ### save the results
        save_txt_path=os.path.join(save_results_path, "RandomizedSearchCV_"+save_classifier_name+".pickle")
        save_pickle(result, save_txt_path)
        save_log("Best parameter for {}: \n result={}.".format(save_classifier_name, result))
        



def evaluate_model(model, X, y):
    """
    Function for evaluating the model.
    """
    score_name='AUC'
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_seed)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1) #, error_score='raise'
    
    return score_name, scores





def main_find_best_model(train_data, feature_columns, keep_feature_columns, label_column, harmonization_settings, save_results_path, feature_selection_type, imbalanced_data_strategy):
    """
    Step 1: Tuning the hyperparameters for different feature selection and classifier models.
    """
    save_log("\n\n == Tuning the hyperparameters for different feature selection and classifier models... ==")
    hyperparameter_tuning_for_different_models(train_data, feature_columns, keep_feature_columns, label_column, harmonization_settings, 
                                               save_results_path, feature_selection_type, imbalanced_data_strategy)
    arrange_hyperparameter_searching_results(save_results_path)


    """
    Step 2: Compare the results of different feature selection and classifier models, with the best tuned hyperparameters.
    """
    save_log("\n\n == Compare the results of different feature selection and classifier models, with the best tuned hyperparameters... ==")
    X=train_data
    y=train_data[label_column]
    best_model_name= explore_different_models(X, y, evaluate_model, save_results_path)
    
    return best_model_name


#======== Step 2: use the best model for training. =========================

def retrain_the_best_model(train_data, label_column, best_model_name, save_results_path):
    """
    Retrain the best model with the whole training dataset. 
    """
    
    save_log("\nWe Retrain the model {} using the whole training dataset. ".format(best_model_name))
    
    train_X=train_data
    train_Y=train_data[label_column]
    
    # fit the model on training data.
    models=get_different_models_from_pickle(save_results_path)
    best_model=models[best_model_name]
    best_model.fit(train_X, train_Y)
    
    # save the trained model.
    save_results_path=os.path.join(save_results_path, best_model_name)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
        
    save_trained_model_path=os.path.join(save_results_path, 'trained_model.sav')
    joblib.dump(best_model, save_trained_model_path)
    
    return save_trained_model_path
    


#=======  Step 3: prediction with the trained best model.========================
 
def predict(trained_model_path, test_data, label_column, save_results_path, data_description, threshold_type="threshold_by_f1"):
    """
    Predict the label with the trained model.
    """
    test_X=test_data
    test_Y=test_data[label_column] if label_column in test_data.columns else None
    
    # txt file to save the chosen threshold for the train data; the chosen threshold will be directly used for test data;
    threshold_file=os.path.join(save_results_path, "thresholds.txt")
    
    #make dir.
    save_results_path=os.path.join(save_results_path, data_description)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
        
    # load the model.
    model = joblib.load(trained_model_path)
    
    # predict.
    if hasattr(model, "predict_proba"):
        predicted_prob = model.predict_proba(test_X)[:,1]
    else:
        predicted_prob = model.predict(test_X)
    
    #save the predicted probability.
    predicted_prob_df=pd.DataFrame(data=predicted_prob, columns=['predicted_prob'], index=test_X.index)
    predicted_prob_df.to_csv(os.path.join(save_results_path, "predicted_prob.csv"), line_terminator='\n')
    
    #calculate the evaluation metrics.
    result_metrics=None
    if test_Y is not None:
        #Precision-Recall curve
        plot_PR_curve(test_Y, predicted_prob, save_results_path)

        #explore the threshold which maximize the f1;
        threshold_by_f1, max_f1=select_threshold_for_binary(test_Y, predicted_prob, save_results_path)
        
        # calculate the threshold which maximize g-means or f1 score.
        if data_description =="train_data":
            # ROC curve with showing the best threshold
            threshold_by_gmeans, max_gmeans=plot_ROC_curve(test_Y, predicted_prob, save_results_path, show_threshold=True)
            
            threshold_info={"data_description":data_description,
                            "threshold_by_gmeans":float(threshold_by_gmeans),
                            "max_gmeans": float(max_gmeans),
                            "threshold_by_f1": float(threshold_by_f1),
                            "max_f1": float(max_f1),
                            "threshold_0.5": 0.5}
            
            save_log("\n Threshold info for train data: \n{}.".format(threshold_info))
            save_dict(threshold_info, threshold_file)
        else:
            # ROC curve without showing the best threshold
            plot_ROC_curve(test_Y, predicted_prob, save_results_path, show_threshold=False)
            
            threshold_info=load_dict(threshold_file)
            save_log("\n Threshold info optimized from the train data: {}.".format(threshold_info))
       
        #the threshold used for make the decision;
        threshold=threshold_info[threshold_type]
        save_log("\n The threshold used: {}.".format(threshold))
            
        #define the threshold
        predicted = predicted_prob > threshold
        predicted_df=pd.DataFrame(data=predicted, columns=['predicted'], index=test_X.index)
        predicted_df.to_csv(os.path.join(save_results_path, "predicted.csv"), line_terminator='\n')

        #plot confusion matrix
        plot_confusion_matrix(test_Y, predicted, save_results_path)
        
        #calculate and save metrics 
        result_metrics=calculate_metrics_for_binary(test_Y, predicted, predicted_prob)
        save_dict({**{"threshold": threshold}, **result_metrics}, os.path.join(save_results_path, "prediction_metrics.txt"))
        save_log("Prediction results:\n{}".format(result_metrics))
        
    return result_metrics
        


#=======================================================================================
"""
Main function for binary classification: train and predict.
"""
def perform_binary_classification_train(train_data, feature_columns, keep_feature_columns, label_column, harmonization_settings,
                                        save_results_path, feature_selection_type, imbalanced_data_strategy):
    """
    Find the best model from a list of models, and retrained it on the whole training dataset.
    """
    save_log("\n\n****** Begin to find and train the best model to predict {} ....... ******".format(label_column))
    
    ## Data preprocessing.
    save_log("\n-train_data.shape={} \n-len(feature_columns)={} \n-label_column={}".format(train_data.shape, len(feature_columns), label_column))

    #Step 1: find the best hyperparameters.
    best_model_name=main_find_best_model(train_data, feature_columns, keep_feature_columns, label_column, harmonization_settings, save_results_path, feature_selection_type, imbalanced_data_strategy)
    #best_model_name="AnovaTest_ExtraTrees"
        
    #Step 2: retrain the selected best model on the whole training dataset.
    trained_model_path=retrain_the_best_model(train_data, label_column, best_model_name, save_results_path)
    
    return best_model_name, trained_model_path

    
def perform_binary_classification_predict(trained_model_path, test_data_dict, label_column, save_results_path):
    """
    make predictions with the trained best model.
    """
    save_results_path=os.path.dirname(trained_model_path)
    
    for description, test_data in test_data_dict.items():
        save_log("\n- Predict for {}: \n-data.shape={}; \n-label_column={};\n-value_count=\n{}".format(description, test_data.shape, label_column, test_data[label_column].value_counts()))

        predict(trained_model_path, test_data, label_column, save_results_path, description)
    




#=======================================================================================
""""
Main: call the function and perform the classification.
"""
def perform_binary_classification(task_name, task_settings, basic_settings):
    print("\n === Basic settings={} =======".format(basic_settings))
        
    save_log("\n =================== task_name={} ===============".format(task_name))
      
    #read the settings
    feature_selection_type=basic_settings["feature_selection_method"]
    imbalanced_data_strategy=basic_settings["imbalanced_data_strategy"]
    
    train_excel_path=task_settings["train_excel_path"]
    test_excel_path_dict=task_settings["test_excel_path_dict"]
    train_data=task_settings["train_data"]
    test_data_dict=task_settings["test_data_dict"]
    feature_columns=task_settings["feature_columns"]
    keep_feature_columns=task_settings["keep_feature_columns"]
    label_column=task_settings["label_column"]
    base_results_path=task_settings["base_results_path"]
    save_log("\n -train_excel_path={}; \n -test_excel_path_dict={}; \n -len(feature_columns)={}; \n -keep_feature_columns={}; \n -label_column={}, \n -base_results_path={}".format(train_excel_path, test_excel_path_dict, len(feature_columns), keep_feature_columns, label_column, base_results_path))

    # create the folder to save results.
    save_results_path=os.path.join(base_results_path, task_name)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path) 
        
#     #drop the highly correlated features using train data.
#     highly_correlated_columns, relatively_indepedent_columns=get_highly_correlated_features(train_data[feature_columns], save_results_path, threshold=0.95)
#     feature_columns=relatively_indepedent_columns
    
    ## Perform ComBat harmonization
    harmonization_settings={
        "harmonization_method": basic_settings["harmonization_method"],
        "harmonization_label": basic_settings["harmonization_label"],
        "harmonization_ref_batch": basic_settings["harmonization_ref_batch"]
    }   
    
    ## -----If using the prediction results of the former classifiers in the classifier chain;-----
    if "former_classifiers" in task_settings.keys():
        use_true_for_train=False
        
        former_classifiers=task_settings["former_classifiers"]
        for former_label, former_task in former_classifiers.items():
            former_task_basepath=os.path.join(base_results_path, former_task)
            former_task_resultfolder=traversalDir_FirstDir(former_task_basepath)[0]
            
            # Train data: use the true/predicted label of the former classifiers in the classifier chain.
            if use_true_for_train:
                train_data[former_label+"_CC"]=train_data[former_label]
            else: 
                former_task_predicted_excel=os.path.join(former_task_basepath, former_task_resultfolder, "train_data", "predicted.csv")
                former_task_predicted_data=pd.read_csv(former_task_predicted_excel, header=0, index_col=0)
                train_data[former_label+"_CC"]=former_task_predicted_data["predicted"].astype('int')
                
            
            # Train data: use the predicted label of the former classifiers in the classifier chain.
            for description, test_data in test_data_dict.items():
                former_task_predicted_excel=os.path.join(former_task_basepath, former_task_resultfolder, description, "predicted.csv")
                former_task_predicted_data=pd.read_csv(former_task_predicted_excel, header=0, index_col=0)
                test_data[former_label+"_CC"]=former_task_predicted_data["predicted"].astype('int')
                test_data_dict[description]=test_data
                
        #add the labels of the former classifiers into the features
        former_classifiers_labels=[former_label+"_CC" for former_label, former_task in former_classifiers.items()]
        keep_feature_columns=keep_feature_columns+former_classifiers_labels    
    #----------------------------------------------------     
    
    ## train the model
    best_model_name, trained_model_path=perform_binary_classification_train(train_data, feature_columns, keep_feature_columns, label_column, harmonization_settings, save_results_path, feature_selection_type, imbalanced_data_strategy)

    ## make predictions
    test_data_dict=dict(**{"train_data":train_data}, **test_data_dict)    
    perform_binary_classification_predict(trained_model_path, test_data_dict, label_column, save_results_path)


    save_log("\nFinish classification for {}!".format(task_name))

