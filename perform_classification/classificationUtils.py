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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
#from sklearn.pipeline import Pipeline
from sklearn import svm, feature_selection

# For evaluation metrics
from sklearn import metrics

#For dealing with imbalanced data
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline 

## feature selection
from probatus.feature_elimination import ShapRFECV

#import function from the self-defined utils
import sys
sys.path.append("../")
from utils.myUtils import mkdir, save_dict, load_dict, get_logger, save_pickle, load_pickle, save_log
from utils.harmonizationUtils import neuroComBat_harmonization, neuroComBat_harmonization_FromTraning

from mySettings import get_basic_settings

# ================= Global variable ==============
global random_seed
random_seed=get_basic_settings()["random_seed"]

# ============ Step 1: find the best model.  ==========================
# 
# - from a list of models, with different feature selection methods and classifiers;
# - using the train data and 5-folds cross-validation.

def hyperparameter_tuning_for_different_models(X, y, save_results_path, feature_selection_type, imbalanced_data_strategy):
    """
    Tuning the hypperparameters for different models.
    """
    ##====== classifiers =======
    classification_models=dict()
    classification_models["SVM"]=svm.SVC()
    classification_models["Perceptron"]=Perceptron()
    classification_models["LogisticRegression"]=LogisticRegression()
    classification_models["GaussianNB"]=GaussianNB()
    classification_models["LinearDiscriminantAnalysis"]=LinearDiscriminantAnalysis()
    classification_models["RandomForest"]=RandomForestClassifier()     
    classification_models["DecisionTree"]=DecisionTreeClassifier()
    classification_models["ExtraTrees"]=ExtraTreesClassifier() 
    #classification_models["LightGBM"]=LGBMClassifier()
    classification_models["XGBClassifier"]=XGBClassifier()
    classification_models["GradientBoosting"]=GradientBoostingClassifier()
    

    ##======  hyperparameters  =======
    param_grids=dict()
    param_grids["SVM"]=[{
        "kernel":["linear", "poly", "rbf", "sigmoid"],
        "C":[0.5, 1, 1.5, 2],
        "gamma":["scale"],
        "class_weight":["balanced"],
        "random_state":[random_seed]
    }]
    
    param_grids["Perceptron"]=[{
        "penalty": ["l1", "l2", "elasticnet", None],
        "alpha":[0.001, 0.0001, 0.00001],
        "class_weight":["balanced"],
        "random_state":[random_seed]
    }]
    
    param_grids["LogisticRegression"]=[
        ## l1 penalty
        {"penalty": ["l1"],
        "C":[0.5, 1, 1.5, 2],
        "solver": ["liblinear", "saga"],
        "class_weight":["balanced"],
        #"max_iter": [500],
        "random_state":[random_seed]},
        ## l2 penalty
        {"penalty": ["l2"], #, 'none'
        "C":[0.5, 1, 1.5, 2],
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        "class_weight":["balanced"],
        #"max_iter": [500],
        "random_state":[random_seed]} 
    ]
    
    param_grids["GaussianNB"]=[{
    }]
        
    param_grids["LinearDiscriminantAnalysis"]=[{
        'solver': ["svd", "lsqr", "eigen"]
    }]
          
    
    param_grids["RandomForest"]=[{
        'n_estimators':  [10, 20, 40, 60, 80, 100],
        'max_features': ['auto', 'sqrt'],
        'max_depth':   [5, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False],
        "random_state":[random_seed]
    }]
    
    param_grids["DecisionTree"]=[{
        "criterion": ["gini", "entropy"],
        "max_depth":  [5, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "random_state":[random_seed],
        "class_weight":["balanced"],
        "random_state":[random_seed]
    }]
        
        
    param_grids["ExtraTrees"]=[{
        "n_estimators":[10, 20, 40, 60, 80, 100],
        "criterion": ["gini", "entropy"],
        'max_depth':  [5, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['auto', 'sqrt', "log2"],
        "class_weight":["balanced", "balanced_subsample"],
        "random_state":[random_seed]
    }]
        
    
    param_grids["LightGBM"]=[{
        "application": ["binary"],
        "boosting": ["gbdt", "rf", "dart", "goss"], 
        #"num_boost_round": [50, 100, 200], 
        "learning_rate": [0.001, 0.01, 0.1],
        "num_leaves": [21, 31, 51], 
        "device": ["gpu"],
        "max_depth":  [5, 10, 20, 30, 40, 50],
        "min_data_in_leaf":  [1, 2, 5, 10, 20],
        "reg_lambda": [0.001, 0.01, 0.1, 0.2, 0.3],
        "verbose": [-1],
        "random_state":[random_seed]
    }]
        
    param_grids["XGBClassifier"]=[{
        "n_estimators":  [10, 20, 40, 60, 80, 100],
        'max_depth':  [5, 10, 20, 30, 40, 50],
        "learning_rate": [0.001, 0.01, 0.1],
        "booster": ["gbtree", "gblinear", "dart"],
        #'min_child_weight': [1, 5, 10],
        #'gamma': [0.5, 1, 2, 5],
        'subsample':  [0.3, 0.7, 1], 
        #'colsample_bytree':  [0, 0.3, 0.7, 1], 
        #'colsample_bylevel':  [0, 0.3, 0.7, 1], 
        'reg_alpha': [0, 1],
        'reg_lambda': [0, 1],
        "use_label_encoder": [False],
        "eval_metric": ["logloss"], 
        "random_state":[random_seed]
    }]
    
    
    param_grids["GradientBoosting"]=[{
        "n_estimators": [10, 20, 40, 60, 80, 100],
        'max_depth':  [5, 10, 20, 30, 40, 50],
        "learning_rate": [0.001, 0.01, 0.1],
        "loss": ["deviance", "exponential"],
        "subsample":  [0.3, 0.7, 1], 
        "criterion": ["friedman_mse", "mse"],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        "random_state":[random_seed]
    }]
    
    ## feature numbers for random search
    feature_number_for_selection=[10, 30, 50, 100]
    
    ## ============ Models =================
    for classfier_name, classifier_model in classification_models.items():
        save_log("\n\n ======Exploring the hyperparameters for feature_selection_method={}, classifier={}. =========".format(feature_selection_type, classfier_name))
        start_time = time()
        
        ### Scaler
        Scaler=StandardScaler()  # MinMaxScaler(feature_range=(0,1))
        #cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        imbalanced_data_handler = get_imbalanced_data_handler(y, imbalanced_data_strategy, random_seed)
        
        #--------------------- begin hyperparameter tuning process--------------------------------
        ### define feature selection function.
        if feature_selection_type=="RFE":
            feature_selection_method=RFE(estimator=classifier_model, step=5) #, n_features_to_select=20
            pipeline = Pipeline(steps=imbalanced_data_handler+[('scaler', Scaler), 
                                       ('feature_selection',feature_selection_method)])
            #save_log("Possible hyperparameters for {} pipeline: \n {}".format(classfier_name, pipeline.get_params().keys()))
            
            param_grid_feature_selection_list=[{"feature_selection__estimator__"+key: item for key, item in param_grid_dict.items()} for param_grid_dict in param_grids[classfier_name]]
            randomsearch_param_grids=[dict(**{"feature_selection__n_features_to_select": feature_number_for_selection}, **param_grid_feature_selection) for param_grid_feature_selection in param_grid_feature_selection_list]
            
            search = RandomizedSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, n_iter=50, scoring="roc_auc", random_state=random_seed, verbose=2).fit(X, y)
            n_feature_selected=search.best_estimator_["feature_selection"].n_features_
            
        elif feature_selection_type=="RFECV": 
            feature_selection_method=RFECV(estimator=classifier_model, step=5, min_features_to_select=20) 
            pipeline = Pipeline(steps=imbalanced_data_handler+[('scaler', Scaler), 
                                       ('feature_selection',feature_selection_method)])
            
            randomsearch_param_grids=[{"feature_selection__estimator__"+key: item for key, item in param_grid_dict.items()} for param_grid_dict in param_grids[classfier_name]]

            search = RandomizedSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, n_iter=50, scoring="roc_auc", random_state=random_seed, verbose=2).fit(X, y)
            n_feature_selected=search.best_estimator_["feature_selection"].n_features_
            
        elif feature_selection_type=="SelectFromModel": 
            feature_selection_method=SelectFromModel(estimator=classifier_model) #max_features=20
            pipeline = Pipeline(steps=imbalanced_data_handler+[('scaler', Scaler), 
                                       ('feature_selection',feature_selection_method)])
            
            param_grid_feature_selection_list=[{"feature_selection__estimator__"+key: item for key, item in param_grid_dict.items()} for param_grid_dict in param_grids[classfier_name]]
            randomsearch_param_grids=[dict(**{"feature_selection__max_features": feature_number_for_selection}, **param_grid_feature_selection) for param_grid_feature_selection in param_grid_feature_selection_list]
            
            search = RandomizedSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, n_iter=50, scoring="roc_auc", random_state=random_seed, verbose=2).fit(X, y)
            n_feature_selected= search.best_estimator_.transform(X).shape[1]
            
        elif feature_selection_type=="AnovaTest" or feature_selection_type=="ChiSquare" or feature_selection_type=="MutualInformation": 
            
            if feature_selection_type=="AnovaTest":
                feature_selection_method=SelectKBest(score_func=f_classif) # k=n_features_to_select

            elif feature_selection_type=="ChiSquare":
                feature_selection_method=SelectKBest(score_func=chi2)

            elif feature_selection_type=="MutualInformation":
                feature_selection_method=SelectKBest(score_func=mutual_info_classif) 
                
            
            pipeline = Pipeline(steps=imbalanced_data_handler+[('scaler', Scaler),  
                                       ('feature_selection',feature_selection_method),
                                       ('classifier',classifier_model)])
            
            param_grid_classifier_list=[{"classifier__"+key: item for key, item in param_grid_dict.items()} for param_grid_dict in param_grids[classfier_name]]
            randomsearch_param_grids=[dict(**{"feature_selection__k": feature_number_for_selection}, **param_grid_classifier) for param_grid_classifier in param_grid_classifier_list]
            
            search = RandomizedSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, n_iter=50, scoring="roc_auc", random_state=random_seed, verbose=1).fit(X, y)
            n_feature_selected=search.best_estimator_["feature_selection"].k
            
        elif feature_selection_type=="PCA":
            feature_selection_method=PCA()
            pipeline = Pipeline(steps=imbalanced_data_handler+[('scaler', Scaler),
                                       ('feature_selection',feature_selection_method),
                                       ('classifier',classifier_model)])

            param_grid_classifier_list=[{"classifier__"+key: item for key, item in param_grid_dict.items()} for param_grid_dict in param_grids[classfier_name]]
            randomsearch_param_grids=[dict(**{"feature_selection__n_components": feature_number_for_selection}, **param_grid_classifier) for param_grid_classifier in param_grid_classifier_list]

            search = RandomizedSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, n_iter=50, scoring="roc_auc", random_state=random_seed, verbose=1).fit(X, y)
            n_feature_selected=search.best_estimator_["feature_selection"].n_components

        else:
            raise Exception("Undefined feature selection function: {} !!".format(feature_selection_type))
        
        
        #--------------------- end hyperparameter tuning process--------------------------------
        
        ### get the best estimator.
        if feature_selection_type=="SelectFromModel":
            best_estimator=Pipeline(steps=imbalanced_data_handler+[('scaler', search.best_estimator_['scaler']), 
                                   ('feature_selection',search.best_estimator_['feature_selection']),
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
        

def get_imbalanced_data_handler(y, imbalanced_data_strategy, random_seed):
    
    counter = Counter(y)
    counter_values=list(counter.values())
    ratio=max(counter_values[0]/counter_values[1], counter_values[1]/counter_values[0])
    
    handler=[]
    if (imbalanced_data_strategy!="IgnoreDataImbalance") and ratio>2:
        if imbalanced_data_strategy=="SMOTE":
            smote=SMOTE(random_state=random_seed, sampling_strategy=0.8) 
            handler=[("sampler", smote)]

        elif imbalanced_data_strategy=="BorderlineSMOTE":
            smote=BorderlineSMOTE(random_state=random_seed, sampling_strategy=0.8)
            handler=[("sampler", smote)]

        elif imbalanced_data_strategy=="SVMSMOTE":
            smote=SVMSMOTE(random_state=random_seed, sampling_strategy=0.8)
            handler=[("sampler", smote)]

        elif imbalanced_data_strategy=="RandomOverSampler":
            OverSampler=RandomOverSampler(random_state=random_seed, sampling_strategy=0.8)
            handler=[("sampler", OverSampler)]

        elif imbalanced_data_strategy=="SMOTE-RandomUnderSampler":
            smote=SMOTE(random_state=random_seed, sampling_strategy=0.4) 
            underSampler = RandomUnderSampler(sampling_strategy=0.8)
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



def explore_different_models(X, y, save_results_path):
    """
    Exploring the models with different feature selection and classifiers, and show the accuracy of these models.
    """

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



def main_find_best_model(train_X, train_Y, save_results_path, feature_selection_type, imbalanced_data_strategy):
    """
    Step 1: Tuning the hyperparameters for different feature selection and classifier models.
    """
    save_log("\n\n == Tuning the hyperparameters for different feature selection and classifier models... ==")
    hyperparameter_tuning_for_different_models(train_X, train_Y, save_results_path, feature_selection_type, imbalanced_data_strategy)
    arrange_hyperparameter_searching_results(save_results_path)


    """
    Step 2: Compare the results of different feature selection and classifier models, with the best tuned hyperparameters.
    """
    save_log("\n\n == Compare the results of different feature selection and classifier models, with the best tuned hyperparameters... ==")
    best_model_name= explore_different_models(train_X, train_Y, save_results_path)
    
    return best_model_name


#======== Step 2: use the best model for training. =========================

def retrain_the_best_model(train_X, train_Y, best_model_name, save_results_path):
    """
    Retrain the best model with the whole training dataset. 
    """
    
    save_log("\nWe Retrain the model {} using the whole training dataset. ".format(best_model_name))
    
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
    # metrics based on the predicted_prob.
    result_metrics["AUC"]=metrics.roc_auc_score(y_true, predicted_prob)
    
    # metrics based on the predicted labels.
    result_metrics["accuracy"]=metrics.accuracy_score(y_true, predicted)
    result_metrics["recall"]=metrics.recall_score(y_true, predicted)
    result_metrics["precision"]=metrics.precision_score(y_true, predicted)
    result_metrics["F1"]=metrics.f1_score(y_true, predicted)
       
    return result_metrics
    
    
def predict(trained_model_path, test_X, test_Y, save_results_path):
    """
    Predict the label with the trained model.
    """
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
        #ROC curve
        plot_ROC_curve(test_Y, predicted_prob, save_results_path)

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
        save_log("Prediction results on test set:\n{}".format(result_metrics))
        
    return result_metrics
        


#=======================================================================================
"""
Main function for binary classification: train and predict.
"""
def perform_binary_classification_train(train_data, feature_columns, label_column, save_results_path, 
                                        feature_selection_type, imbalanced_data_strategy):
    """
    Find the best model from a list of models, and retrained it on the whole training dataset.
    """
    save_log("\n\n****** Begin to find and train the best model to predict {} ....... ******".format(label_column))
    
    ## Data preprocessing.
    train_X=train_data[feature_columns]
    train_Y=train_data[label_column]
    save_log("\n-train_data.shape={} \n-len(feature_columns)={} \n-label_column={}".format(train_data.shape, len(feature_columns), label_column))

    #Step 1: find the best hyperparameters.
    best_model_name=main_find_best_model(train_X, train_Y, save_results_path, feature_selection_type, imbalanced_data_strategy)
    #best_model_name="AnovaTest_ExtraTrees"
        
    #Step 2: retrain the selected best model on the whole training dataset.
    trained_model_path=retrain_the_best_model(train_X, train_Y, best_model_name, save_results_path)
    
    return best_model_name, trained_model_path

    
def perform_binary_classification_predict(trained_model_path, test_data_dict, feature_columns, label_column, save_results_path):
    """
    make predictions with the trained best model.
    """
    save_results_path=os.path.dirname(trained_model_path)
    
    for description, test_data in test_data_dict.items():
        save_log("\n- Predict for {}: \n-test_data.shape={} \n-len(feature_columns)={} \n-label_column={}".format(description, test_data.shape, len(feature_columns), label_column))
        test_X=test_data[feature_columns]
        test_Y=test_data[label_column] if label_column in test_data.columns else None

        predict(trained_model_path, test_X, test_Y, os.path.join(save_results_path, description))
    

#=======================================================================================
"""
Function: perform ComBat harmonzation.
"""
def perform_harmonization(train_data, test_data_dict, feature_columns, harmonization_method, harmonization_label, harmonization_ref_batch):
    print("\n\n Begin to harmonize the data....")
    print("\n harmonization_method={}, number of features={}, harmonization_ref_batch={}.".format(harmonization_method, len(feature_columns), harmonization_ref_batch))
        
    # harmonization for the train data, and learn the estimates used for test data.
    print("\n Available harmonization label for train data: \n {}.".format(train_data[harmonization_label].value_counts()))
    train_setting_label_list=np.unique(train_data[harmonization_label])
    harmonized_train_data, estimates, info=neuroComBat_harmonization(train_data, feature_columns, harmonization_label, harmonization_method, harmonization_ref_batch)
    
    #print("\n estimates['batches']={}".format(estimates['batches']))
    #print("\n info={}".format(info))
    
    # harmonize the test data using the learnt estimates.
    harmonized_test_data_dict={}
    for description, test_data in test_data_dict.items():
        print("\n Available harmonization label for {}: \n {}.".format(description, test_data[harmonization_label].value_counts()))
        
        #delete the rows whose setting labels are not in the training data.
        test_setting_label_list=np.unique(test_data[harmonization_label])
        abnormal_setting_label_list= list(set(test_setting_label_list).difference(set(train_setting_label_list)))
        #print("\n train_setting_label_list={}.".format(train_setting_label_list))
        #print("\n test_setting_label_list={}.".format(test_setting_label_list))
        #print("\n abnormal_setting_label_list={}.\n".format(abnormal_setting_label_list))
        if len(abnormal_setting_label_list)>0:
            warnings.warn("Warning: will delete the data with setting labels {}, which do not exist in training dataset!!".format(abnormal_setting_label_list))
            for abnormal_setting_label in abnormal_setting_label_list:
                test_data.drop(test_data[test_data[harmonization_label]==abnormal_setting_label].index, inplace=True)
        
        #harmomize the test data;
        harmonized_test_data, estimates_test=neuroComBat_harmonization_FromTraning(test_data, feature_columns, harmonization_label, estimates)
        harmonized_test_data_dict[description]=harmonized_test_data
        
    return harmonized_train_data, harmonized_test_data_dict

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
    relatively_indepedent_columns= list(set(original_feature_columns).difference(set(highly_correlated_columns)))
    
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
    label_column=task_settings["label_column"]
    base_results_path=task_settings["base_results_path"]
    save_log("\n -train_excel_path={}; \n -test_excel_path_dict={}; \n -len(feature_columns)={}; \n -label_column={}, \n -base_results_path={}".format(train_excel_path, test_excel_path_dict, len(feature_columns), label_column, base_results_path))

    # create the folder to save results.
    save_results_path=os.path.join(base_results_path, task_name)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path) 
        
    #drop the highly correlated features using train data.
    highly_correlated_columns, relatively_indepedent_columns=get_highly_correlated_features(train_data, feature_columns, save_results_path, threshold=0.95)
    feature_columns=relatively_indepedent_columns
    
    ## Perform ComBat harmonization
    harmonization_method=basic_settings["harmonization_method"]
    harmonization_label=basic_settings["harmonization_label"]
    harmonization_ref_batch=basic_settings["harmonization_ref_batch"]
    if harmonization_method!="withoutComBat":
        if harmonization_label=="is_3T":
            modality_list=basic_settings["feature_filter_dict"]["modality_list"]
            print("\n modality_list={}".format(modality_list))
            for modality in modality_list:
                harmonization_label_for_modality=harmonization_label+"_"+modality
                
                #filter the features for this modality
                feature_columns_for_modality=[]
                for feature in feature_columns:
                    if feature.startswith(modality):
                        feature_columns_for_modality.append(feature)
                        
                #harmonization for this modality data
                if len(feature_columns_for_modality)>0:
                    train_data, test_data_dict=perform_harmonization(train_data, test_data_dict, feature_columns_for_modality, 
                                                                 harmonization_method, harmonization_label_for_modality, 
                                                                 harmonization_ref_batch)
        else:
            train_data, test_data_dict=perform_harmonization(train_data, test_data_dict, feature_columns, 
                                                             harmonization_method, harmonization_label, harmonization_ref_batch)
    
    
    ## train the model
    best_model_name, trained_model_path=perform_binary_classification_train(train_data, feature_columns, label_column, save_results_path, feature_selection_type, imbalanced_data_strategy)

    ## make predictions
    test_data_dict=dict(**{"train_data":train_data}, **test_data_dict)    
    perform_binary_classification_predict(trained_model_path, test_data_dict, feature_columns, label_column, save_results_path)


    save_log("\nFinish classification for {}!".format(task_name))

