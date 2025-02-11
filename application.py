import pickle
from flask import Flask, request, render_template,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

#Route for  a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint_dl():
    if request.method == "GET":
        return render_template('index.html')
    else:
        data = CustomData(
            SeniorCitizen=request.form.get('SeniorCitizen'),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            tenure=int(request.form.get('tenure')),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            InternetService=request.form.get('InternetService'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            TotalCharges=float(request.form.get('TotalCharges'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print("pred_df:\n",pred_df)
        predict_pipeline = PredictPipeline()
        
        results = predict_pipeline.predict(pred_df)
        
        print("Predicted Maths score",results)
        
        return render_template('index.html',results=results[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)