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
from joblib import dump, load
model_path = 'kmeans_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
dump(model, 'kmeans_model.joblib')
# model = load('kmeans_model.joblib')
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


app = Flask(__name__, template_folder='templates')



model = pickle.load(open('kmeans_model.pkl', 'rb'))
template_dir = os.path.abspath('templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Enable template auto-reloading
app.config['STATIC_FOLDER'] = os.path.abspath('static')



def load_and_clean_data(file_path):
    retail = pd. read_csv (file_path, sep=",", encoding="ISO-8859-1", header=0)
    retail [ 'CustomerID'] = retail [ 'CustomerID'].astype (str)
    
    retail['Amount'] = retail['Quantity']*retail['UnitPrice']
    rfm=retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID') [ 'InvoiceNo'] .count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%m/%d/%Y %H:%M')

    max_date= max(retail ['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID') [ 'Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm = pd.merge(rfm, rfm_f, on='CustomerID', how= 'inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    
    # Convert 'Amount', 'Frequency', and 'Recency' to numeric with error handling
     # Convert 'Amount', 'Frequency', and 'Recency' to numeric with error handling
    numeric_columns = ['Amount', 'Frequency', 'Recency']
    try:
        rfm[numeric_columns] = rfm[numeric_columns].apply(pd.to_numeric, errors='coerce')
    except pd.errors.OverflowError as e:
        # Handle overflow errors if necessary
        print(f"Error converting columns to numeric: {e}")
        return None

    # Drop rows with NaN values in any of the numeric columns
    rfm = rfm.dropna(subset=numeric_columns)

    # Make sure the conversion was successful
    if rfm[numeric_columns].dtypes.any() == 'object':
        print("Error: Conversion to numeric failed.")
        return None


    Q1 = rfm.quantile(0.05)
    Q3 = rfm.quantile(0.95)
    IQR = Q3 - Q1

    rfm = rfm[(rfm.Amount >= Q1['Amount'] - 1.5 * IQR['Amount']) & (rfm.Amount <= Q3['Amount'] + 1.5 * IQR['Amount'])]
    rfm = rfm[(rfm.Recency >= Q1['Recency'] - 1.5 * IQR['Recency']) & (rfm.Recency <= Q3['Recency'] + 1.5 * IQR['Recency'])]
    rfm = rfm[(rfm.Frequency >= Q1['Frequency'] - 1.5 * IQR['Frequency']) & (rfm.Frequency <= Q3['Frequency'] + 1.5 * IQR['Frequency'])]

    return rfm


def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame (rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
    return rfm,rfm_df_scaled;



@app.route('/')
def home():
    template_path = os.path.join(app.root_path, app.template_folder, 'index.html')
    print(f"Template Path: {template_path}")
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
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
    response={'amount_img':amount_img_path,
              'freq_img':freq_img_path,
              'recency_img':recency_img_path}
    return json.dumps(response)


if __name__=="__main__" :
    app.run(debug=True)

