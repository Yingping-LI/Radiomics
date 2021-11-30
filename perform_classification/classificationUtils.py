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
import operator
import joblib
import warnings
from collections import Counter

## For plots
import seaborn as sns
import matplotlib.pyplot as plt

## For preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# For feature selection and classifier models
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, LassoCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFECV, RFE, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm, feature_selection

# For evaluation metrics
from sklearn import metrics

#For dealing with imbalanced data
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import Pipeline 

## feature selection
from probatus.feature_elimination import ShapRFECV

## data imputation
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion

#import function from the self-defined utils
import sys
sys.path.append("../")
from utils.myUtils import mkdir, save_dict, load_dict, get_logger, save_pickle, load_pickle, save_log
from utils.harmonizationUtils import neuroComBat_harmonization, neuroComBat_harmonization_FromTraning
from MyTransformers import ComBatTransformer, PandasSimpleImputer, SelectColumnsTransformer, DeleteCorrColumnTransformer

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
    
    ##====== classifiers =======
    classification_models=dict()
    classification_models["SVM"]=svm.SVC()
    classification_models["Perceptron"]=Perceptron()
    classification_models["LogisticRegression"]=LogisticRegression()
#     classification_models["GaussianNB"]=GaussianNB()
#     classification_models["LinearDiscriminantAnalysis"]=LinearDiscriminantAnalysis()
    classification_models["RandomForest"]=RandomForestClassifier()     
    classification_models["DecisionTree"]=DecisionTreeClassifier()
    classification_models["ExtraTrees"]=ExtraTreesClassifier() 
    #classification_models["LightGBM"]=LGBMClassifier()
    classification_models["XGBClassifier"]=XGBClassifier()
    classification_models["GradientBoosting"]=GradientBoostingClassifier()
    

    ##======  hyperparameters  =======
    param_grids=dict()
    param_grids["SVM"]={
        "kernel":["linear", "poly", "rbf", "sigmoid"],
        "C":[0.01, 0.1, 1, 10, 100],
        #"class_weight":["balanced"],
        "random_state":[np.random.RandomState(random_seed)]
    }
    
    param_grids["Perceptron"]={
        "penalty": ["l1", "l2", "elasticnet", None],
        "alpha":[0.1, 0.01, 0.001, 0.0001, 0.00001],
        #"class_weight":["balanced"],
        "random_state":[np.random.RandomState(random_seed)]
    }
    
    param_grids["LogisticRegression"]={
#         ## l1 penalty
#         "penalty": ["l1"],
#         "C":[0.5, 1, 1.5, 2],
#         "solver": ["liblinear", "saga"],
#         "class_weight":["balanced"],
#         #"max_iter": [500],
#         "random_state":[random_seed]},
        ## l2 penalty
        "penalty": ["l2"], 
        "C":[0.01, 0.1, 1, 10, 100],
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        #"max_iter": [500],
        #"class_weight":["balanced"],
        "random_state":[np.random.RandomState(random_seed)]} 
    
    
#     param_grids["GaussianNB"]={
#     }
        
#     param_grids["LinearDiscriminantAnalysis"]={
#         'solver': ["svd", "lsqr", "eigen"]
#     }
          
    
    param_grids["RandomForest"]={
        "criterion": ["gini", "entropy"],
         #'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators':  [10, 20, 50, 100],
        'max_depth':   [5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'bootstrap': [True, False],
        "random_state":[np.random.RandomState(random_seed)]
    }
    
    param_grids["DecisionTree"]={
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        #"class_weight":["balanced"],
        "random_state":[np.random.RandomState(random_seed)]
    }
        
        
    param_grids["ExtraTrees"]={
        "criterion": ["gini", "entropy"],
        #'max_features': ['auto', 'sqrt', "log2"],
        "n_estimators":[10, 20, 50, 100],
        'max_depth':  [5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        #"class_weight":["balanced", "balanced_subsample"],
        'bootstrap': [True, False],
        "random_state":[np.random.RandomState(random_seed)]
    }
        
    
    param_grids["LightGBM"]={
        "application": ["binary"],
        "boosting": ["gbdt", "rf", "dart", "goss"], 
        #"num_boost_round": [50, 100, 200], 
        "learning_rate": [0.001, 0.01, 0.1],
        "num_leaves": [11, 21, 31, 51], 
        #"device": ["gpu"],
        "max_depth":  [5, 10, 15, 20, 30],
        "min_data_in_leaf": [1, 2, 5],
        #"reg_lambda": [0.001, 0.01, 0.1, 0.2, 0.3],
        #"verbose": [-1],
        "random_state":[np.random.RandomState(random_seed)]
    }
        
    param_grids["XGBClassifier"]={
        "n_estimators": [10, 20, 50, 100],
        'max_depth':  [5, 10, 15, 20, 30],
        "learning_rate": [0.001, 0.01, 0.1],
        "booster": ["gbtree", "gblinear", "dart"],
        'min_child_weight': [1, 2, 5],
        #'gamma': [0.5, 1, 2, 5],
        #'subsample':  [0.3, 0.7, 1], 
        #'colsample_bytree':  [0, 0.3, 0.7, 1], 
        #'colsample_bylevel':  [0, 0.3, 0.7, 1], 
        #'reg_alpha': [0, 1],
        #'reg_lambda': [0, 1],
        #"use_label_encoder": [False],
        #"eval_metric": ["logloss"], 
        "random_state":[np.random.RandomState(random_seed)]
    }
    
    
    param_grids["GradientBoosting"]={
        "criterion": ["friedman_mse", "mse"],
        #'max_features': ['auto', 'sqrt', 'log2'],
        "n_estimators": [10, 20, 50, 100],
        'max_depth':  [5, 10, 15, 20, 30],
        "learning_rate": [0.001, 0.01, 0.1],
        #"loss": ["deviance", "exponential"],
        #"subsample":  [0.3, 0.7, 1], 
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        "random_state":[np.random.RandomState(random_seed)]
    }
    
    ## feature numbers for random search
    feature_number_for_selection=[20, 40, 60, 80, 100]
    
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
        delete_corr_features_transformer=[("del_corr_features", DeleteCorrColumnTransformer(threshold=0.95))]
        
        ##-----
        preprocessing_transformer_list=imputation_transformer+harmonization_transformer+[
            ("filter_features", SelectColumnsTransformer(feature_columns))]+imbalanced_data_handler+scaler_transformer+delete_corr_features_transformer
        
        cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        
        #--------------------- begin hyperparameter tuning process--------------------------------
        ### define feature selection function.
        if feature_selection_type=="RFE":
            feature_selection_method=RFE(estimator=classifier_model, step=5) #, n_features_to_select=20
            pipeline = Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            #save_log("Possible hyperparameters for {} pipeline: \n {}".format(classfier_name, pipeline.get_params().keys()))
            
            randomsearch_param_grids=dict(**{"feature_selection__n_features_to_select": feature_number_for_selection}, **{"feature_selection__estimator__"+key: item for key, item in param_grids[classfier_name].items()}) 
            
            search = searchCV(pipeline, randomsearch_param_grids, cross_val, random_seed).fit(X, y)
            n_feature_selected=search.best_estimator_["feature_selection"].n_features_
            
        elif feature_selection_type=="RFECV": 
            feature_selection_method=RFECV(estimator=classifier_model, step=5, min_features_to_select=20) 
            pipeline = Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            
            randomsearch_param_grids={"feature_selection__estimator__"+key: item for key, item in param_grids[classfier_name].items()}

            search = searchCV(pipeline, randomsearch_param_grids, cross_val, random_seed).fit(X, y)
            n_feature_selected=search.best_estimator_["feature_selection"].n_features_
            
        elif feature_selection_type=="SelectFromModel": 
            feature_selection_method=SelectFromModel(estimator=classifier_model) #max_features=20
            pipeline = Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            
           
            randomsearch_param_grids=dict(**{"feature_selection__max_features": feature_number_for_selection}, **{"feature_selection__estimator__"+key: item for key, item in param_grids[classfier_name].items()}) 
            
            search = searchCV(pipeline, randomsearch_param_grids, cross_val, random_seed).fit(X, y)
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
            pipeline = Pipeline(steps=[('features',combined_features), ('classifier',classifier_model)])           
            
            # random search parameters
            randomsearch_param_grids=dict(**{"features__selected_features__feature_selection__k": feature_number_for_selection}, **{"classifier__"+key: item for key, item in param_grids[classfier_name].items()}) 
            
            # grid search
            search =searchCV(pipeline, randomsearch_param_grids, cross_val, random_seed, searchCV_method).fit(X, y)
            n_feature_selected=search.best_estimator_["features"].get_params()["selected_features"]["feature_selection"].k
            
        elif feature_selection_type=="PCA":
            feature_selection_method=PCA()
            
            # Pipeline
            selected_features=Pipeline(steps=preprocessing_transformer_list+[('feature_selection',feature_selection_method)])
            combined_features = FeatureUnion([("keep_features_directly", SelectColumnsTransformer(keep_feature_columns)), ("selected_features", selected_features)])
            pipeline = Pipeline(steps=[('features',combined_features), ('classifier',classifier_model)]) 
            
            # random search parameters
            randomsearch_param_grids=dict(**{"features__selected_features__feature_selection__n_components": feature_number_for_selection}, **{"classifier__"+key: item for key, item in param_grids[classfier_name].items()})

            search = searchCV(pipeline, randomsearch_param_grids, cross_val, random_seed).fit(X, y)
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
        

def searchCV(pipeline, randomsearch_param_grids, cross_val, random_seed, searchCV_method):
    """
    Choose from the gridSearchCV and randomSearchCV;
    """
    if searchCV_method=="gridSearchCV":
        SearchCV_= GridSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, scoring="roc_auc", verbose=1, 
                                return_train_score=True, error_score='raise')
        
    elif searchCV_method=="randomSearchCV":
        SearchCV_= RandomizedSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, n_iter=50, scoring="roc_auc", random_state=random_seed, verbose=1, return_train_score=True)
        
    return SearchCV_

        
def plot_GridSearch_results(grid, save_image_path=None):
    """
    Plot the train/validation performance with regards to different hyperparameters. 
    Note that, when tuning one hyperparameters, fixed the other hyperparameters to their best values achieved by GridSearch.
    
    Params: 
        grid: A trained GridSearchCV object.
        save_image_path: where to save the plots.
        
    See references:https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    """
        
    ## Results from grid search.
    cv_results = grid.cv_results_
    mean_train_score = cv_results['mean_train_score']
    std_train_score = cv_results['std_train_score']
    mean_test_score = cv_results['mean_test_score']
    std_test_score = cv_results['std_test_score']

    ## All considered parameters and their values;
    parameter_names= list(grid.best_params_.keys())
    parameter_grids=grid.param_grid
    
    ## Getting indexes of values per hyper-parameter
    masks=[]
    for parameter_key, parameter_value in grid.best_params_.items():
        masks.append(list(cv_results['param_'+parameter_key].data==parameter_value))

    ## Ploting results
    fig, ax = plt.subplots(1, len(parameter_grids), sharex='none', sharey='all', figsize=(20,5))
    #fig.suptitle('AUC by differet hyperparameters!')
    fig.text(0.04, 0.5, 'Mean AUC', va='center', rotation='vertical')
    for i, param in enumerate(parameter_names):
        if param!="classifier__random_state":
            # get the index of the experiments who have set the best values of the other hyperparameters except this specific parameter "param".
            masks_without_this_param = np.stack(masks[:i] + masks[i+1:])
            best_experiments_masks= masks_without_this_param.all(axis=0)
            best_experiments_index = np.where(best_experiments_masks)[0]

            # get the results of these exeperiments and plot them.
            x = np.array(parameter_grids[param])
            y_train = np.array(mean_train_score[best_experiments_index])
            e_train = np.array(std_train_score[best_experiments_index])
            y_test = np.array(mean_test_score[best_experiments_index])
            e_test = np.array(std_test_score[best_experiments_index])

            ax[i].errorbar(x, y=y_train, yerr=e_train, linestyle='-', marker='o',label='train' )
            ax[i].errorbar(x, y=y_test, yerr=e_test, linestyle='--', marker='o', label='test')
            ax[i].set_xlabel(param.upper())

    plt.legend()
    plt.savefig(save_image_path)
    plt.show()
    
    
def get_imbalanced_data_handler(y, imbalanced_data_strategy, random_seed):
    
    counter = Counter(y)
    counter_values=list(counter.values())
    ratio=max(counter_values[0]/counter_values[1], counter_values[1]/counter_values[0])
    
    handler=[]
    if (imbalanced_data_strategy!="IgnoreDataImbalance") and ratio>2:
        if imbalanced_data_strategy=="SMOTE":
            smote=SMOTE(random_state=random_seed, sampling_strategy="auto") 
            handler=[("sampler", smote)]

        elif imbalanced_data_strategy=="BorderlineSMOTE":
            smote=BorderlineSMOTE(random_state=random_seed, sampling_strategy="auto")
            handler=[("sampler", smote)]

        elif imbalanced_data_strategy=="SVMSMOTE":
            smote=SVMSMOTE(random_state=random_seed, sampling_strategy="auto")
            handler=[("sampler", smote)]

        elif imbalanced_data_strategy=="RandomOverSampler":
            OverSampler=RandomOverSampler(random_state=random_seed, sampling_strategy="auto")
            handler=[("sampler", OverSampler)]
            
        elif imbalanced_data_strategy=="RandomUnderSampler":
            underSampler=RandomUnderSampler(random_state=random_seed, sampling_strategy="auto")
            handler=[("sampler", underSampler)]

        elif imbalanced_data_strategy=="SMOTE-RandomUnderSampler":
            smote=SMOTE(random_state=random_seed, sampling_strategy=0.4) 
            underSampler = RandomUnderSampler(random_state=random_seed, sampling_strategy=0.8)
            handler=[("smote", smote), ("underSampler", underSampler)]

        else:
            raise Exception("Undefined strategy for dealing with imblanced data. Possible strategy: \{\"SMOTE\", \"BorderlineSMOTE\", \"SVMSMOTE\", \"RandomOverSampler\", \"SMOTE_RandomUnderSampler\"\}.")
    
    save_log("\nImbalanced data strategy={}, Data counter={}, imbalanced data handler={}.".format(imbalanced_data_strategy, counter, handler))
    
    return handler




def get_all_classifier_list():
    """
    List of the models considered for comparison.
    """
    
    feature_selection_method_list=["RFE", "RFECV", "AnovaTest", "ChiSquare", "MutualInformation", "SelectFromModel", "PCA"]
    classifier_list=["SVM", "Perceptron", "LogisticRegression", "GaussianNB", "LinearDiscriminantAnalysis",
                     "RandomForest", "DecisionTree", "ExtraTrees", "LightGBM", "GradientBoosting", "XGBClassifier"]
    
    classifiers=[]
    for feature_selection_type in feature_selection_method_list:
        for classfier_name in classifier_list:
            classifiers.append(feature_selection_type+"_"+classfier_name)
            
    return classifiers



def arrange_hyperparameter_searching_results(results_path):   
    """
    Arrange the results of all different classifiers, after performing best hyperparameter searching for each classifier.
    """
    
    classifiers=get_all_classifier_list()
    
    Results=[]
    for classfier_name in classifiers:
        hyperparameter_result_path=os.path.join(results_path, "RandomizedSearchCV_"+classfier_name+".pickle")
        if os.path.exists(hyperparameter_result_path):
            hyperparameter_result=load_pickle(hyperparameter_result_path)
            Results.append(hyperparameter_result)
        
    ## Sorting results by best score
    Results = sorted(Results, key=operator.itemgetter('best score'), reverse=True)
    save_pickle(Results, os.path.join(results_path, "RandomizedSearchCV_all_models.pickle"))
    save_log("\n\n ******RandomizedSearchCV Results: *****\n{}".format(Results))
    
    ## best classifier
    best_classifier_name=Results[0]["classfier_name"]
    
    return best_classifier_name


# In[ ]:


def evaluate_model(model, X, y):
    """
    Function for evaluating the model.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_seed)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, error_score='raise') 
    
    return scores

def get_different_models_from_pickle(model_basepath):
    """
    Define the models from the pickle files which saves the best hyperparameters trainded by RandomSearchCV.
    """
    
    save_log("\n Load the model paramters from {}.".format(model_basepath))
    
    ## read the models from the pickle files.
    models=dict()
    classifiers=get_all_classifier_list()
    for classfier_name in classifiers:
        hyperparameter_result_path=os.path.join(model_basepath, "RandomizedSearchCV_"+classfier_name+".pickle")
        if os.path.exists(hyperparameter_result_path):
            hyperparameter_result=load_pickle(hyperparameter_result_path)
            best_estimator=hyperparameter_result["best_estimator"]
            models[classfier_name]=best_estimator
            save_log("\n Use the best hyperparameter found by RandomSearchCV  for {}: {}.".format(classfier_name, best_estimator))

    return models



def explore_different_models(train_data, label_column, save_results_path):
    """
    Exploring the models with different feature selection and classifiers, and show the accuracy of these models.
    """

    X=train_data
    y=train_data[label_column]
    
    # get the models to evaluate
    models = get_different_models_from_pickle(save_results_path)
    
    # evaluate the models and save the results
    results=[]
    for model_name, model in models.items():
        start_time = time()
        scores = evaluate_model(model, X, y)
        time_cost=time()-start_time
        results.append((model_name, np.median(scores), np.mean(scores), np.std(scores), time_cost, scores))
        save_log('> %s: median_score= %.3f , mean_score= %.3f , std_score= %.3f, time=%.2f seconds.' % (model_name, np.median(scores), np.mean(scores), np.std(scores), time_cost))
    
    results=pd.DataFrame(results, columns=["model_name", "median_AUC", "mean_AUC", "std_AUC", "Time(seconds)", "AUC"])
    results.sort_values("mean_AUC", ascending=False, inplace=True)
    results.to_csv(os.path.join(save_results_path, "AUC_results_all_models.txt"))
    save_log("\n\n ***********rank average of the AUC scores: ************\n{}".format(results))
    
    # plot model performance for comparison
    plt.subplots(figsize=(15,5))
    plt.boxplot(results["AUC"], labels=results["model_name"], showmeans=True)
    plt.xlabel('Feature selection and classifier models', fontsize=15)
    plt.ylabel('AUC',fontsize=15)
    plt.xticks(rotation=15)
    plt.subplots_adjust(left=0.05, bottom=0.25, right=0.95, top=0.95, wspace =0, hspace =0) 
    save_fig_path=os.path.join(save_results_path, "explore_different_models.jpeg")
    plt.savefig(save_fig_path)
    plt.show()
    
    # best classifier
    best_classifier_name=results.iloc[0]["model_name"]

    return best_classifier_name



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
    best_model_name= explore_different_models(train_data, label_column, save_results_path)
    
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


def plot_ROC_curve(y_true, predicted_prob, save_results_path):
    """
    Plot the ROC curve.
    """
    
    # calculate the fpr/tpr values and AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, predicted_prob)
    save_log("thresholds={}".format(thresholds))
    roc_auc_score = metrics.auc(fpr, tpr)  
    
    #plot the ROC curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=3, label='area = %0.2f' % roc_auc_score)
    ax.plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax.set(xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver Operating Characteristic")     
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.savefig(os.path.join(save_results_path, "ROC_curve.jpeg"))         
    plt.show()
    
    
def plot_PR_curve(y_true, predicted_prob, save_results_path):
    """
    Plot the Precision-Recall curve.
    """
    # calculate the precision, recall values and AUC
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, predicted_prob)
    save_log("thresholds={}".format(thresholds))
    auc_PR = metrics.auc(recalls, precisions)
    
    #plot the precision-recall curve.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(recalls, precisions, color='darkorange', lw=3, label='area = %0.2f' % auc_PR)
    ax.set(xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")     
    ax.legend(loc="lower left")
    ax.grid(True)
    plt.savefig(os.path.join(save_results_path, "PR_curve.jpeg"))         
    plt.show()

    
def select_threshold(y_true, predicted_prob, save_results_path):
    
    #Calculate the evaluation metrics.
    metrics_dict={"accuracy":[], "precision":[], "recall":[], "F1":[]}
    thresholds=np.arange(0.1, 1, step=0.1)
    for threshold in thresholds:
        predicted= predicted_prob>threshold
        
        metrics_dict["accuracy"].append(metrics.accuracy_score(y_true, predicted))
        metrics_dict["recall"].append(metrics.recall_score(y_true, predicted))
        metrics_dict["precision"].append(metrics.precision_score(y_true, predicted))
        metrics_dict["F1"].append(metrics.f1_score(y_true, predicted))
    
    metrics_df=pd.DataFrame(metrics_dict).set_index(pd.Index(thresholds))
    
    # plot the values of these metrics depending on different threshold
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    metrics_df.plot(ax=ax)
    ax.set(xlabel='threshold', ylabel="metrics", title="Threshold Selection")
    ax.legend(loc="lower left")
    ax.grid(True)
    plt.savefig(os.path.join(save_results_path, "threshold_selection.jpeg"))  
    plt.show()
    
    # Choose the threshold which maximize the F1-score, if F1 is equal, then choose the one which maximizes the accuracy.
    metrics_df.sort_values(by=["F1", "accuracy"],  ascending=[False, False], inplace=True)
    best_threshold=metrics_df.index[0]

    return best_threshold
 
    
def plot_confusion_matrix(y_true, predicted, save_results_path):
    """
    Plot the confusion matrix.
    """
    classes = np.unique(y_true)
    cm = metrics.confusion_matrix(y_true, predicted, labels=classes)
    
    # plot the confusion matrix.
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax.set_yticklabels(labels=classes)
    plt.savefig(os.path.join(save_results_path, "confusion_matrix.jpeg"))
    plt.show()
    
    
def calculate_metrics(y_true, predicted, predicted_prob):
    """
    Calcualte the metrics for evaluation.
    """
    result_metrics={}
    ## metrics based on the predicted_prob.
    result_metrics["AUC_ROC"]=metrics.roc_auc_score(y_true, predicted_prob)
    result_metrics["AveragePrecisions"]=metrics.average_precision_score(y_true, predicted_prob)
    
    #Use AUC function to calculate the area under the curve of precision recall curve.
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, predicted_prob)
    result_metrics["AUC_PR"]=metrics.auc(recalls, precisions)
    
    ## metrics based on the predicted labels.
    result_metrics["accuracy"]=metrics.accuracy_score(y_true, predicted)
    result_metrics["balanced_accuracy"]=metrics.balanced_accuracy_score(y_true, predicted)
    result_metrics["recall"]=metrics.recall_score(y_true, predicted)
    result_metrics["precision"]=metrics.precision_score(y_true, predicted)
    result_metrics["F1"]=metrics.f1_score(y_true, predicted)
       
    return result_metrics
    
    
def predict(trained_model_path, test_data, label_column, save_results_path):
    """
    Predict the label with the trained model.
    """
    test_X=test_data
    test_Y=test_data[label_column] if label_column in test_data.columns else None
    
    #make dir.
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
        #ROC curve and Precision-Recall curve
        plot_ROC_curve(test_Y, predicted_prob, save_results_path)
        plot_PR_curve(test_Y, predicted_prob, save_results_path)

        #explore the threshold
        threshold=select_threshold(test_Y, predicted_prob, save_results_path)
        save_log("\nThe threshold which maximize the F1 score is: {}.".format(threshold))

        #define the threshold
        predicted = predicted_prob > threshold

        #plot confusion matrix
        plot_confusion_matrix(test_Y, predicted, save_results_path)
        
        #calculate and save metrics 
        result_metrics=calculate_metrics(test_Y, predicted, predicted_prob)
        save_dict(result_metrics, os.path.join(save_results_path, "prediction_metrics.txt"))
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

        predict(trained_model_path, test_data, label_column, os.path.join(save_results_path, description))
    

#=======================================================================================
def get_highly_correlated_features(dataframe, original_feature_columns, save_results_path=None, threshold=0.95):
    """
    Split the feature columns to two classes:
    - highly_correlated_columns: features which has a correlation coefficient > 0.95 (threshold) with one of the other features.
    - relatively_indepedent_columns: the other features which are not in the highly correlated feature list.
    
    See more here:
    https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
    """
    
    features=dataframe[original_feature_columns]
      
    #save the correlation matrix plot
    if save_results_path is not None:
        with plt.style.context({'axes.labelsize':24,
                        'xtick.labelsize':24,
                        'ytick.labelsize':24}):
            plt.subplots(figsize=(100, 80))
            sns.heatmap(features.corr(), annot=False, cmap='YlGnBu')
            plt.tight_layout()
            plt.savefig(os.path.join(save_results_path, 'feature_correlation_matrix.jpeg'))
            plt.show()
        
    #Calculate the upper trigular matrix.    
    cor_matrix = features.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    
    #Calculate the highly correlated feature columns and the relatively indepent feature columns.
    highly_correlated_columns = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Notes!!! the order will be changed randomly when convert "list" to "set", so we sort the final list for reproducible results.
    relatively_indepedent_columns= list(set(original_feature_columns).difference(set(highly_correlated_columns)))
    relatively_indepedent_columns=sorted(relatively_indepedent_columns)
    
    save_log("\n\nAmong the {} original features: \n-{} features are highly corelated with other features, so will be dropped. \n-{} features are kept and can be regarded as relatively indepent features.".format(len(original_feature_columns), len(highly_correlated_columns), len(relatively_indepedent_columns)))
    
    return highly_correlated_columns, relatively_indepedent_columns


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
#     highly_correlated_columns, relatively_indepedent_columns=get_highly_correlated_features(train_data, feature_columns, save_results_path, threshold=0.95)
#     feature_columns=relatively_indepedent_columns
    
    ## Perform ComBat harmonization
    harmonization_settings={
        "harmonization_method": basic_settings["harmonization_method"],
        "harmonization_label": basic_settings["harmonization_label"],
        "harmonization_ref_batch": basic_settings["harmonization_ref_batch"]
    }   
    
    ## train the model
    best_model_name, trained_model_path=perform_binary_classification_train(train_data, feature_columns, keep_feature_columns, label_column, harmonization_settings, save_results_path, feature_selection_type, imbalanced_data_strategy)

    ## make predictions
    test_data_dict=dict(**{"train_data":train_data}, **test_data_dict)    
    perform_binary_classification_predict(trained_model_path, test_data_dict, label_column, save_results_path)


    save_log("\nFinish classification for {}!".format(task_name))

