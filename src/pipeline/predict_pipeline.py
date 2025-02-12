import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        print("features:\n",features)
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            print("preds inside predict function",preds)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(self, SeniorCitizen:str,Partner:str,Dependents:str,tenure:int,
                 InternetService:str,OnlineSecurity:str,OnlineBackup:str,DeviceProtection:str,
                 TechSupport:str,Contract:str,PaperlessBilling:str,PaymentMethod:str,
                 TotalCharges:str):
        self.customerID = 'ABC-123'
        self.gender = 'Male'
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = 'Yes'
        self.MultipleLines = 'No'
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = 'Yes'
        self.StreamingMovies = 'Yes'
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = '33.33'
        self.TotalCharges = TotalCharges
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "customerID" : [self.customerID],
                "gender" : [self.gender],
                "SeniorCitizen" : [str(self.SeniorCitizen)],
                "Partner" : [self.Partner],
                "Dependents" : [self.Dependents],
                "tenure" : [self.tenure],
                "PhoneService" : [self.PhoneService],
                "MultipleLines" : [self.MultipleLines],
                "InternetService" : [self.InternetService],
                "OnlineSecurity" : [self.OnlineSecurity],
                "OnlineBackup" : [self.OnlineBackup],
                "DeviceProtection" : [self.DeviceProtection],
                "TechSupport" : [self.TechSupport],
                "StreamingTV" : [self.StreamingTV],
                "StreamingMovies" : [self.StreamingMovies],
                "Contract" : [self.Contract],
                "PaperlessBilling" : [self.PaperlessBilling],
                "PaymentMethod" : [self.PaymentMethod],
                "MonthlyCharges" : [self.MonthlyCharges],
                "TotalCharges" : [self.TotalCharges]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)