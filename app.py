from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
app = Flask(__name__)
model = pickle.load(open('kmeans_model.pkl', 'rb'))
def load_and_clean_data(file_path):
    retail = pd. read_csv (file_path, sep=",", encoding="ISO-8859-1", header=0)
    retail [ 'Customer ID'] = retail [ 'Customer ID'].astype (str)
    retail['Amount'] = retail['Quantity']*retail['UnitPrice']
    rfm=retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('Customer ID') [ 'InvoiceNo'] .count().reset_index()
    rfm_f.columns = ['Customer ID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%d- %m-%Y %H:%M')
    max_date= max(retail ['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('Customer ID') [ 'Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how= 'inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    Q1 = rfm. quantile (0.05)
    Q3 = rfm.quantile (0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm. Amount >= Q1[0] - 1.5* IQR [0]) & (rfm.Amount <= Q3 [0] +1.5*IQR[0])]
    rfm = rfm[(rfm. Recency >= Q1[2] - 1.5*IQR [2]) & (rfm. Recency <= Q3[2] + 1.5*IQR[2])]
    rfm = rfm[(rfm. Frequency >= Q1[1] - 1.5* IQR [1]) & (rfm. Frequency <= Q3[1] + 1.5* IQR [1] )]
    return rfm
def preprocess_data (file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame (rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
    return rfm,rfm_df_scaled;
@app.route('/')
def home():
    return render_template('index.html')
@app.route('predict',methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os. path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df = preprocess_data(file_path) [1] 
    results_df = model.predict(df) 
    df_with_id = preprocess_data(file_path) [0]
    df_with_id['Cluster_Id'] = results_df
    sns. stripplot (x='Cluster_Id', y='Amount', data=df_with_id, hue='Cluster_Id')
    amount_img_path = 'static/ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()
    sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id, hue='Cluster_Id')
    freq_img_path = 'static/Cluster Id_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()
    sns.stripplot (x='Cluster_Id', y='Recency', data=df_with_id,hue='Cluster_Id')
    recency_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()
    response={'amount_img':amount_img_path,'freq_img':freq_img_path,'recency_img':recency_img_path}
    return json.dumps(response)
if __name__=="__main__" :
    app.run(debug=True)

