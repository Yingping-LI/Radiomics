#!/usr/bin/env python
# coding: utf-8

import os
import operator
import joblib
import numpy as np
import pandas as pd
from numpy import argmax
from time import time
from numpy import sqrt
from collections import Counter
from scipy import interp


# classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, LassoCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn import svm



#For dealing with imbalanced data
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# For gridsearch and randomsearch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# For evaluation metrics
from sklearn import metrics
from sklearn.metrics import classification_report

## For plots
import seaborn as sns
import matplotlib.pyplot as plt

#import self-defined functions
import sys
sys.path.append("../")
from utils.myUtils import save_log, save_pickle, load_pickle, mkdir


#=======================================================


def get_classifiers(random_seed):
    """
    Dict of different classifiers and their hyperparameters;
    """
    
    classification_models=dict()
    classification_models["SVM"]=svm.SVC()
    classification_models["Perceptron"]=Perceptron()
    classification_models["LogisticRegression"]=LogisticRegression()
#     classification_models["KNeighborsClassifier"]=KNeighborsClassifier()
#     classification_models["GaussianNB"]=GaussianNB()
#     classification_models["LinearDiscriminantAnalysis"]=LinearDiscriminantAnalysis()
    classification_models["RandomForest"]=RandomForestClassifier()     
    classification_models["DecisionTree"]=DecisionTreeClassifier()
    classification_models["ExtraTrees"]=ExtraTreesClassifier() 
#     #classification_models["LightGBM"]=LGBMClassifier()
#     classification_models["XGBClassifier"]=XGBClassifier()
    classification_models["GradientBoosting"]=GradientBoostingClassifier()
#     classification_models["MLPClassifier"]=MLPClassifier()
    

    ##======  hyperparameters  =======
    param_grids=dict()
    param_grids["SVM"]={
        "kernel":["linear", "poly", "rbf", "sigmoid"],
        "C":[0.01, 0.1, 0.5, 1, 5, 10, 100],
        "probability": [True], 
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
        "C":[0.01, 0.1, 0.5, 1, 5, 10, 100],
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        "max_iter": [500],
        #"class_weight":["balanced"],
        "random_state":[np.random.RandomState(random_seed)]} 
    
    param_grids["KNeighborsClassifier"]={
        "n_neighbors":[5, 10],
        "weights":["uniform", "distance"],
        "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
        "p": [1, 2]
    }
        
    param_grids["GaussianNB"]={
    }
        
    param_grids["LinearDiscriminantAnalysis"]={
        'solver': ["svd", "lsqr", "eigen"]
    }
          
    
    param_grids["RandomForest"]={
        "criterion": ["gini", "entropy"],
         #'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators':  [10, 20, 50, 100],
        'max_depth':   [5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        #"class_weight":["balanced", "balanced_subsample"],
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
        #"is_unbalance":[True]
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
        "criterion": ["friedman_mse", "squared_error"],
        #'max_features': ['auto', 'sqrt', 'log2'],
        "n_estimators": [10, 20, 50, 100],
        'max_depth':  [5, 10, 15, 20, 30],
        "learning_rate": [0.001, 0.01, 0.1],
        "loss": ["deviance", "exponential"],
        "subsample":  [0.3, 0.7, 1], 
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        "random_state":[np.random.RandomState(random_seed)]
    }
    
    
    param_grids["MLPClassifier"]={
        "hidden_layer_sizes": [(100, 50, 20), (100, 50, 50, 20, 20), [50]*5, [100]*5, [100]*5+[50]*5],
        "activation": ["relu"],
        'solver':  ["sgd", "adam"],
        "alpha": [0.0001, 0.001],
        "batch_size": [5, 10],
        "learning_rate": ["constant", "invscaling", "adaptive"], 
        #'learning_rate_init': [0.001],
        #'max_iter': [200],
        #"tol": [1e-4],
        #"momentum":[0.9],
        "early_stopping": [True],
        #"n_iter_no_change": [10],
        "random_state":[np.random.RandomState(random_seed)]
    }
    
    return classification_models, param_grids
   


def get_all_classifier_list():
    """
    List of the models considered for comparison.
    """
    
    feature_selection_method_list=["RFE", "RFECV", "AnovaTest", "ChiSquare", "MutualInformation", "SelectFromModel", "PCA"]
    classifier_list=["SVM", "Perceptron", "LogisticRegression", "KNeighborsClassifier", "GaussianNB", "LinearDiscriminantAnalysis",
                     "RandomForest", "DecisionTree", "ExtraTrees", "LightGBM", "GradientBoosting", "XGBClassifier", "MLPClassifier"]
    
    classifiers=[]
    for feature_selection_type in feature_selection_method_list:
        for classfier_name in classifier_list:
            classifiers.append(feature_selection_type+"_"+classfier_name)
            
    return classifiers



def get_imbalanced_data_handler(y, imbalanced_data_strategy, random_seed):
    """
    For binary classification, choose the imbalance data strategy if the ratio of the y bigger 2.
    """
    
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

        elif imbalanced_data_strategy=="SMOTE_RandomUnderSampler":
            smote=SMOTE(random_state=random_seed, sampling_strategy=0.4) 
            underSampler = RandomUnderSampler(random_state=random_seed, sampling_strategy=0.8)
            handler=[("smote", smote), ("underSampler", underSampler)]

        else:
            raise Exception("Undefined strategy for dealing with imblanced data. Possible strategy: \{\"SMOTE\", \"BorderlineSMOTE\", \"SVMSMOTE\", \"RandomOverSampler\", \"SMOTE_RandomUnderSampler\"\}.")
    
    save_log("\nImbalanced data strategy={}, Data counter={}, imbalanced data handler={}.".format(imbalanced_data_strategy, counter, handler))
    
    return handler




def plot_ROC_curve(y_true, predicted_prob, save_results_path, show_threshold):
    """
    Plot the ROC curve.
    """
    
    # calculate the fpr/tpr values and AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, predicted_prob)
    save_log("thresholds={}".format(thresholds))
    roc_auc_score = metrics.auc(fpr, tpr)  
    
    # calculate the threshold which maximize g-mean value;
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    threshold=thresholds[ix]
    max_gmeans=gmeans[ix]
    save_log('Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    
    #plot the ROC curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=3, label='area = %0.2f' % roc_auc_score)
    ax.plot([0,1], [0,1], color='navy', lw=3, linestyle='--', label='No Skill')
    if show_threshold:
        ax.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    ax.set(xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver Operating Characteristic")     
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.savefig(os.path.join(save_results_path, "ROC_curve.jpeg"))         
    plt.show()
    
    return threshold, max_gmeans


def plot_ROC_curve_for_multilabel(y_true, y_predicted_prob, label_names, save_results_path, show_threshold=False):
    """
    Plot the ROC curve for multilabel classification.
    """
    
    num_labels=len(label_names)
    fig, axes = plt.subplots(1, num_labels, figsize=(6*num_labels, 6))
    
    for i in range(num_labels):
        ax=axes.flatten()[i]
        label=label_names[i]
        y_true_i=y_true[:, i]
        y_pred_i=y_predicted_prob[:, i]
        
        # calculate the fpr/tpr values and AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true_i, y_pred_i)
        roc_auc_score = metrics.auc(fpr, tpr)  
        
        # calculate the threshold which maximize g-mean value;
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        threshold=thresholds[ix]
        max_gmeans=gmeans[ix]
        save_log('Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    
        #plot the ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=3, label='area = %0.2f' % roc_auc_score)
        ax.plot([0,1], [0,1], color='navy', lw=3, linestyle='--', label='No Skill')
        if show_threshold:
            ax.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        ax.set(xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", 
               title="Receiver Operating Characteristic( " + str(label)+" )")     
        ax.legend(loc="lower right")
        ax.grid(True)
        
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)
    plt.savefig(os.path.join(save_results_path, "ROC_curve.jpeg"))        
    plt.show()
    

    
def plot_all_ROC_curves_for_multilabel(y_true, y_predicted_prob, label_names, save_results_path):
    """
    Plot the ROC curve for multilabel classification.
    
    See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    """
    
    fpr_dict=dict()
    tpr_dict=dict()
    roc_auc_dict=dict()
    
    ## --- calcuate for each bianry classification problem. -------
    num_labels=len(label_names)
    for i in range(num_labels):
        label=label_names[i]
        y_true_i=y_true[:, i]
        y_pred_i=y_predicted_prob[:, i]
        
        # calculate the fpr/tpr values and AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true_i, y_pred_i)
        roc_auc_score = metrics.auc(fpr, tpr)  
        
        # save the fpr/tpr/AUC in the dict.
        fpr_dict[label]=fpr
        tpr_dict[label]=tpr
        roc_auc_dict[label]=roc_auc_score
   

    ## --- calculate the macro averaage. ----------
    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[key] for key in fpr_dict.keys()]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for key in fpr_dict.keys():
        mean_tpr += interp(all_fpr, fpr_dict[key], tpr_dict[key])
        
    mean_tpr /= num_labels

    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = metrics.auc(fpr_dict["macro"], tpr_dict["macro"])
    
    ## ---- calculate the micro average. --------
    fpr_dict["micro"], tpr_dict["micro"], _ = metrics.roc_curve(y_true.ravel(), y_predicted_prob.ravel())
    roc_auc_dict["micro"] = metrics.auc(fpr_dict["micro"], tpr_dict["micro"]) 
    
    ## ----- Plot the ROC curve. -------------
    
    plt.figure(figsize=(8, 6))
    i=0
    colors=["aqua", "darkorange", "cornflowerblue", "deeppink", "navy"]
    for label, auc in roc_auc_dict.items():
        fpr=fpr_dict[label]
        tpr=tpr_dict[label]
        if label in ["micro", "macro"]:
            plt.plot(fpr, tpr,  color=colors[i],  linestyle=":",  lw=4, label=label+"-average ROC curve (area = {0:0.2f})".format(roc_auc_dict[label]))
        else:
            plt.plot(fpr, tpr,  color=colors[i],  lw=2,  label="ROC curve for {0} (area = {1:0.2f})".format(label, roc_auc_dict[label]))
            
        
        i=i+1
        
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curves")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_results_path, "ROC_curves_all.jpeg"))  
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
    
    
def plot_confusion_matrix(y_true, predicted, save_results_path, save_file_name="confusion_matrix.jpeg"):
    """
    Plot the confusion matrix.
    """
    classes = np.unique(y_true)
    cm = metrics.confusion_matrix(y_true, predicted, labels=classes)
    
    # plot the confusion matrix.
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax.set_xticklabels(labels=classes)
    ax.set_yticklabels(labels=classes)
    plt.savefig(os.path.join(save_results_path, save_file_name))
    plt.show()
    
    
def plot_confusion_matrix_for_multilabel(y_true, y_predicted, label_names, save_results_path):
    """
    Plot the confusion matrix for multi-label classification problem.
    """
    
    cm = metrics.multilabel_confusion_matrix(y_true, y_predicted)
    print("cm={}".format(cm))
    
    num_labels=len(label_names)
    fig, ax = plt.subplots(1, num_labels, figsize=(6*num_labels, 6))
    
    for axes, label, cm_i in zip(ax.flatten(), label_names, cm):
        heatmap=sns.heatmap(cm_i, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False, ax=axes)
        axes.set_ylabel('True')
        axes.set_xlabel('Predicted')
        axes.set_title("Confusion Matrix ( " + str(label)+" )")  
        
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)
    plt.savefig(os.path.join(save_results_path, "confusion_matrix.jpeg"))
    plt.show()
    
    
def calculate_metrics_for_binary(y_true, predicted, predicted_prob):
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

def calculate_metrics_for_multilabel(y_true, y_predicted, y_predicted_prob, label_names, save_results_path):
    """
    Calcualte the metrics for evaluation.
    """
    
    ## Calculate the metrics for each binary problem.
    binary_result_metrics={}
    num_labels=len(label_names) 
    for i in range(num_labels):
        label=label_names[i]
        y_true_i=y_true[:, i]
        y_pred_i=y_predicted[:, i] 
        y_pred_prob_i=y_predicted_prob[:, i] 
        binary_result_metrics[label]=calculate_metrics_for_binary(y_true_i, y_pred_i, y_pred_prob_i)
        
    binary_result_metrics_df=pd.DataFrame(binary_result_metrics).transpose()  
    save_log("\n\n-Binary prediction metrics:\n{}".format(binary_result_metrics_df))
      
    ## Calcualte the classification reports;
    overall_classification_report=classification_report(y_true, y_predicted, target_names=label_names, output_dict=True)
    classification_report_df=pd.DataFrame(overall_classification_report).transpose()
    save_log("\n\n-Classification report:\n{}".format(classification_report_df))
    
    ## save the results
    excel_writer = pd.ExcelWriter(os.path.join(save_results_path, "prediction_metrics.xlsx"))
    binary_result_metrics_df.to_excel(excel_writer,sheet_name='binary_result_metrics')
    classification_report_df.to_excel(excel_writer,sheet_name='classification_report')
    excel_writer.save()  

    ## Calculate some other overall metrics for the multi-label problem
    result_metrics={}
    result_metrics["overall_accuracy"]=metrics.accuracy_score(y_true, y_predicted)
        
    return result_metrics

def calculate_metrics_for_multiclass(y_true, y_predicted):
    """
    Calcualte the metrics for multiclass classification problem.
    """
    
    result_metrics={}
    result_metrics["accuracy"]=metrics.accuracy_score(y_true, y_predicted)

    return result_metrics

def get_highly_correlated_features(feature_df, save_results_path=None, threshold=0.95):
    """
    Split the feature columns to two classes:
    - highly_correlated_columns: features which has a correlation coefficient > 0.95 (threshold) with one of the other features.
    - relatively_indepedent_columns: the other features which are not in the highly correlated feature list.
    
    See more here:
    https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
    """
    
    original_feature_columns=feature_df.columns
      
    #save the correlation matrix plot
    if save_results_path is not None:
        with plt.style.context({'axes.labelsize':24,
                        'xtick.labelsize':24,
                        'ytick.labelsize':24}):
            plt.subplots(figsize=(100, 80))
            sns.heatmap(feature_df.corr(), annot=False, cmap='YlGnBu')
            plt.tight_layout()
            plt.savefig(os.path.join(save_results_path, 'feature_correlation_matrix.jpeg'))
            plt.show()
        
    #Calculate the upper trigular matrix.    
    cor_matrix = feature_df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    
    #Calculate the highly correlated feature columns and the relatively indepent feature columns.
    highly_correlated_columns = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Notes!!! the order will be changed randomly when convert "list" to "set", so we sort the final list for reproducible results.
    relatively_indepedent_columns= list(set(original_feature_columns).difference(set(highly_correlated_columns)))
    relatively_indepedent_columns=sorted(relatively_indepedent_columns)
    
    save_log("\n\nAmong the {} original features: \n-{} features are highly corelated with other features, so will be dropped. \n-{} features are kept and can be regarded as relatively indepent features.".format(len(original_feature_columns), len(highly_correlated_columns), len(relatively_indepedent_columns)))
    
    return highly_correlated_columns, relatively_indepedent_columns




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


def searchCV(pipeline, randomsearch_param_grids, scoring, cross_val, random_seed, searchCV_method):
    """
    Choose from the gridSearchCV and randomSearchCV;
    """
    if searchCV_method=="gridSearchCV":
        SearchCV_= GridSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, scoring=scoring, verbose=1, 
                                return_train_score=True, error_score='raise')
        
    elif searchCV_method=="randomSearchCV":
        SearchCV_= RandomizedSearchCV(pipeline, randomsearch_param_grids, cv=cross_val, n_iter=50, scoring=scoring, random_state=random_seed, verbose=1, return_train_score=True)
        
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
    fig.text(0.04, 0.5, 'Mean Score', va='center', rotation='vertical')
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
    
    
    
    
def select_threshold_for_binary(y_true, predicted_prob, save_results_path):
    """
    Select threshold which maximize the F1-score.
    """
    
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
    max_f1=metrics_df.iloc[0]["F1"]

    return best_threshold, max_f1


def select_threshold_for_multilabel(y_true, predicted_prob, save_results_path, average='macro'):
    """
    Select threshold which maximize the F1-score.
    
    Note: average is in {None, 'micro', 'macro', 'weighted', 'samples'}.
    """
    
    #Calculate the evaluation metrics.
    metrics_dict={"accuracy":[], "precision":[], "recall":[], "F1":[]}
    thresholds=np.arange(0.1, 1, step=0.1)
    for threshold in thresholds:
        predicted= predicted_prob>threshold
        
        metrics_dict["accuracy"].append(metrics.accuracy_score(y_true, predicted))
        metrics_dict["recall"].append(metrics.recall_score(y_true, predicted, average=average))
        metrics_dict["precision"].append(metrics.precision_score(y_true, predicted, average=average))
        metrics_dict["F1"].append(metrics.f1_score(y_true, predicted, average=average))
    
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
    max_f1=metrics_df.iloc[0]["F1"]

    return best_threshold, max_f1


def explore_different_models(X, y, evaluate_model_function, save_results_path):
    """
    Exploring the models with different feature selection and classifiers, and show the accuracy of these models.
    """

    # get the models to evaluate
    models = get_different_models_from_pickle(save_results_path)
    
    # evaluate the models and save the results
    results=[]
    for model_name, model in models.items():
        start_time = time()
        score_name, scores = evaluate_model_function(model, X, y)

        # delete nan values;
        scores=[x for x in scores if str(x) != 'nan']
        
        time_cost=time()-start_time
        results.append((model_name, np.median(scores), np.mean(scores), np.std(scores), time_cost, scores))
        save_log('> %s: median_score= %.3f , mean_score= %.3f , std_score= %.3f, time=%.2f seconds.' % (model_name, np.median(scores), np.mean(scores), np.std(scores), time_cost))
    
    results=pd.DataFrame(results, columns=["model_name", "median_"+score_name, "mean_"+score_name, "std_"+score_name, "Time(seconds)", score_name])
    results.sort_values("mean_"+score_name, ascending=False, inplace=True)
    results.to_csv(os.path.join(save_results_path, score_name+"_results_all_models.txt"))
    save_log("\n\n ***********rank average of the {} scores: ************\n{}".format(score_name, results))
    
    # plot model performance for comparison
    plt.subplots(figsize=(15,5))
    plt.boxplot(results[score_name], labels=results["model_name"], showmeans=True)
    plt.xlabel('Feature selection and classifier models', fontsize=15)
    plt.ylabel(score_name,fontsize=15)
    plt.xticks(rotation=15)
    plt.subplots_adjust(left=0.05, bottom=0.25, right=0.95, top=0.95, wspace =0, hspace =0) 
    save_fig_path=os.path.join(save_results_path, "explore_different_models.jpeg")
    plt.savefig(save_fig_path)
    plt.show()
    
    # best classifier
    best_classifier_name=results.iloc[0]["model_name"]

    return best_classifier_name
