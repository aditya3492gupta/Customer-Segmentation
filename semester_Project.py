#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle


# In[2]:


retail = pd.read_csv('G:\Code\cus\data.csv', sep=',', encoding='ISO-8859-1', header=0)

retail.head()


# In[3]:


retail.shape


# In[4]:


retail.info()


# In[5]:


df_null = round (100* (retail.isnull(). sum())/len (retail), 2)
df_null


# In[6]:


retail = retail.dropna()
retail.shape


# In[7]:


retail['CustomerID'] = retail['CustomerID'].astype(str)


# In[8]:


#monetary 
retail['Amount'] = retail['Quantity'] *retail['UnitPrice']
rfm_m = retail.groupby('CustomerID') ['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()


# In[9]:


# frequency
rfm_f = retail.groupby('CustomerID') ['InvoiceNo'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
rfm_f.head()


# In[10]:


rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how= 'inner')
rfm.head()


# In[14]:


retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%m/%d/%Y %H:%M')
retail['InvoiceDate']


# In[15]:


max_date= max(retail['InvoiceDate'])
max_date


# In[16]:


retail [ 'Diff'] = max_date - retail['InvoiceDate']
retail.head()


# In[17]:


rfm_p = retail.groupby('CustomerID')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p.head()


# In[18]:


# recency
rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()


# In[19]:


rfm = pd.merge(rfm, rfm_p, on='CustomerID', how= 'inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
rfm.head()


# In[20]:


attributes = ['Amount', 'Frequency', 'Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot (data = rfm[attributes], orient="", palette="Set2",whis=1.5, saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold') 
plt.xlabel("Attributes", fontweight = 'bold')


# In[21]:


Q1 = rfm. Amount.quantile (0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm. Amount >= Q1 - 1.5*IQR) & (rfm. Amount <= Q3 + 1.5*IQR)]


# In[22]:


Q1 = rfm. Recency.quantile(0.05)
Q3 = rfm. Recency.quantile (0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm. Recency >= Q1 - 1.5*IQR) & (rfm. Recency <= Q3 + 1.5*IQR)]


# In[23]:


Q1 = rfm. Frequency.quantile (0.05)
Q3 = rfm. Frequency.quantile (0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm. Frequency >= Q1 - 1.5*IQR) & (rfm. Frequency <= Q3 +1.5*IQR)]


# In[24]:


rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
scaler = StandardScaler()
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[25]:


rfm_df_scaled = pd. DataFrame (rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']


# In[26]:


kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit (rfm_df_scaled)


# In[27]:


kmeans.labels_


# In[28]:


set(kmeans.labels_)


# In[29]:


ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8] 
for num_clusters in range_n_clusters:
    kmeans = KMeans (n_clusters=num_clusters, max_iter=50)
    kmeans.fit (rfm_df_scaled)
    ssd.append (kmeans. inertia_)
plt.plot(ssd)


# In[30]:


kmeans = KMeans(n_clusters=3, max_iter=300)
kmeans.fit(rfm_df_scaled)


# In[31]:


kmeans.labels_


# In[32]:


kmeans.predict(rfm_df_scaled)


# In[33]:


rfm[ 'Cluster_Id'] = kmeans.predict(rfm_df_scaled)
rfm.head()


# In[34]:


filename= 'kmeans_model.pkl'
with open('kmeans_saved_model', 'wb') as file:
    pickle.dump (kmeans, file)
file.close()
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))


# In[35]:


kmeans.labels_


# In[37]:


rfm['Cluster_Id']=kmeans.predict(rfm_df_scaled)
rfm.head()


# In[38]:


sns.stripplot(x='Cluster_Id',y='Amount',data=rfm)


# In[40]:


sns.stripplot(x='Cluster_Id',y='Recency',data=rfm)
plt.savefig("Cluster_IdRecency.png")


# In[ ]:
sns.boxplot(data=rfm[attributes], orient='v', palette='Set2', whis=1.5, saturation=1, width=0.7)


sns.stripplot(x='Cluster_Id', y='Amount', data=rfm)
plt.savefig("Cluster_Id_Amount.png")
plt.clf()  # Add this line to clear the current figure for the next plot

sns.stripplot(x='Cluster_Id', y='Recency', data=rfm)
plt.savefig("Cluster_Id_Recency.png")
plt.clf()  # Add this line to clear the current figure for the next plot



