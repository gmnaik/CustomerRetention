import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def conversion_datatype_columns(self,train_df,test_df):
        train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
        test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
        
        # Drop NaN values if exists
        train_df = train_df.dropna(subset=['TotalCharges'])
        test_df = test_df.dropna(subset=['TotalCharges'])
        
        #Convert Senior Citizen values from Yes and No to 1 and 0 so that we can encode all columns at once
        mapping = {0: 'No', 1: 'Yes'}
        train_df['SeniorCitizen'] = train_df['SeniorCitizen'].map(mapping)
        test_df['SeniorCitizen'] = test_df['SeniorCitizen'].map(mapping)
        
        #Convert datatype of SeniorCitizen and TotalCharges
        train_df['SeniorCitizen'] = train_df['SeniorCitizen'].astype(object)
        train_df['TotalCharges'] = train_df['TotalCharges'].astype(float)
        
        test_df['SeniorCitizen'] = test_df['SeniorCitizen'].astype(object)
        test_df['TotalCharges'] = test_df['TotalCharges'].astype(float)
        
        #Drop columns from train and test dataframe as they dont play a significant role in churn prediction according to EDA
        train_df = train_df.drop(columns=['customerID','gender','PhoneService','StreamingTV','StreamingMovies','MultipleLines','MonthlyCharges'])
        test_df = test_df.drop(columns=['customerID','gender','PhoneService','StreamingTV','StreamingMovies','MultipleLines','MonthlyCharges'])
        
        #Obtain numeric and categorical columns present in dataframe.
        numeric_feature = [feature for feature in train_df.columns if train_df[feature].dtype != 'O']
        categorical_feature = [feature for feature in train_df.columns if train_df[feature].dtype == 'O']
        
        return train_df,test_df,numeric_feature, categorical_feature
    
    def get_data_transformer_object(self,numerical_columns,categorical_columns):
        '''
        This function is responsible for data transformation
        '''
        try:       
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
                )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("label_encoder", OneHotEncoder())
                ]
                )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            logging.info("Conversion of datatype columns has started")
            
            train_df,test_df,numerical_columns,categorical_columns = self.conversion_datatype_columns(train_df,test_df)
            
            logging.info("Split dependent and independent variables")
            X_train = train_df.drop('Churn',axis = 1)
            y_train = train_df['Churn']
            
            X_test = test_df.drop('Churn',axis = 1)
            y_test = test_df['Churn']
            
            logging.info("Oversampling using RandomOverSampler")
            oversample = RandomOverSampler(sampling_strategy='minority')
            X_over, y_over = oversample.fit_resample(X_train, y_train)

            logging.info("Obtaining preprocessing object")
            
            #Drop 'Churn' from categorical_columns list as it is dependent column
            categorical_columns.remove("Churn")
            
            preprocessing_obj = self.get_data_transformer_object(numerical_columns,categorical_columns)
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            X_train = preprocessing_obj.fit_transform(X_over)
            X_test = preprocessing_obj.transform(X_test)
            
            mapping_y = {'No': 0, 'Yes': 1}
            y_train = y_over.map(mapping_y)
            y_test = y_test.map(mapping_y)

            train_arr = np.c_[X_train,np.array(y_train)]
            test_arr = np.c_[X_test,np.array(y_test)]
            
            logging.info("Saved preprocessing object.")
            
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            