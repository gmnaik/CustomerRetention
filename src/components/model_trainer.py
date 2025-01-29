import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_object

from sklearn.metrics import roc_curve, roc_auc_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def evaluate_model(self,true, predicted):
        accuracyscore = accuracy_score(true, predicted)
        classificationreport = classification_report(true, predicted)
        classificationreport_dict = classification_report(true, predicted, output_dict=True)
        confusionmatrix = confusion_matrix(true, predicted)
        return accuracyscore, classificationreport,classificationreport_dict, confusionmatrix
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "RandomForest Classifier": RandomForestClassifier(random_state=42),
                "LGBM Classifier": lgb.LGBMClassifier(random_state=42),
                "CatBoost Classifier": CatBoostClassifier(verbose=0, random_state=42), 
                "SVM Classifier": SVC(random_state=42),
                "KNeighbors Classifier": KNeighborsClassifier() 
            }
            param_grids = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'solver': ['saga'],
                    'class_weight' : ['balanced'],
                },
                "Linear Discriminant Analysis": {
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]  # Only valid for 'lsqr' or 'eigen'
                },
                "Decision Tree Classifier": {
                    'criterion': ['gini', 'entropy'],
                    'class_weight' : ['balanced'],
                    'max_depth': [3, 5, 10,12],
                    'min_samples_split': [1, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                    'ccp_alpha': [0.0, 0.01, 0.1]
                },
                "RandomForest Classifier": {
                    'n_estimators': [20,30,50, 100],
                    'max_depth': [3, 5,11],
                    'min_samples_split': [1, 5, 11],
                    'min_samples_leaf': [1, 2, 5],
                    'class_weight' : ['balanced', 'balanced_subsample']
                },
                
                "LGBM Classifier": {
                    'n_estimators': [50, 70],
                    'num_leaves': [31, 50],
                    'max_depth': [10, 20],
                    'learning_rate': [0.01, 0.1],
                    'is_unbalance': [True, False],  
                    'scale_pos_weight': [1, 10]
                },
                "CatBoost Classifier": {
                    'iterations': [200, 230],
                    'depth': [6, 9],
                    'learning_rate': [0.1,0.2],
                    'class_weights': [[1, 5], [1, 10]] 
                },
                
                "SVM Classifier": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced']
                },
                "KNeighbors Classifier": {
                    'n_neighbors': [7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
                
            }
            
            model_list = []
            f1score_list =[]
            
            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = param_grids[list(models.keys())[i]]
                #model.fit(X_train, y_train) # Train model
                
                gs = GridSearchCV(model,para,cv=3,scoring='f1_weighted', n_jobs=-1)
                gs.fit(X_train,y_train)
                
                #model = gs.best_estimator_
                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Evaluate Train and Test dataset
                model_train_accuracyscore , model_train_classificationreport,classificationreport_train_dict,model_train_confusionmatrix  = self.evaluate_model(y_train, y_train_pred)

                model_test_accuracyscore , model_test_classificationreport,classificationreport_test_dict,model_test_confusionmatrix = self.evaluate_model(y_test, y_test_pred)

                
                print(list(models.keys())[i])
                model_list.append(list(models.keys())[i])
                print('Model performance for Training set')
                print("- Accuracy score: \n{:.4f}".format(model_train_accuracyscore))
                print("- Classification report:\n {}".format(model_train_classificationreport))

                print('----------------------------------')
                
                print('Model performance for Test set')
                print("- Accuracy score: \n{:.4f}".format(model_test_accuracyscore))
                print("- Classification report: \n{}".format(model_test_classificationreport))
               
                f1score_list.append(classificationreport_test_dict['weighted avg']['f1-score'])
            
            model_dict = {}

            for i in range(0,len(model_list)):
                model_dict[model_list[i]] = f1score_list[i]

            best_model_score = max(sorted(model_dict.values()))
            
            #To get best model name from dictionary
            best_model_name = list(model_dict.keys())[list(model_dict.values()).index(best_model_score)]
                        
            best_model = models[best_model_name]

            print("model_dict:",model_dict)
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            else:
                pass
            
            logging.info("Best model found")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model) 
            
            y_pred_prob = best_model.predict_proba(X_test)[:, 1] 

            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            auc_score = roc_auc_score(y_test, y_pred_prob)
            
            print(f"AUC: {auc_score:.2f}")
            #print("fpr:",fpr)
            #print("tpr:",tpr)
            print("thresholds:",thresholds)
            
            optimal_idx = np.argmax(tpr-fpr)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal Threshold: {optimal_threshold:.2f}")
            
            #predicted = best_model.predict(X_test)
            
            #r2_square = r2_score(y_test,predicted)
            
            return best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)