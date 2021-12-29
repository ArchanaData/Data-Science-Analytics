#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[10]:


tf.config.experimental.list_physical_devices()


# In[14]:


df = pd.read_csv('Telco_cus_churn.csv')


# In[15]:


df.head()


# In[16]:


df.Churn.value_counts()


# In[17]:


517400/df.shape[0]


# In[18]:


df.drop('customerID',axis='columns',inplace=True)


# In[19]:


df.dtypes


# In[20]:


df.TotalCharges.values


# In[21]:


pd.to_numeric(df.TotalCharges,errors='coerce').isnull()


# In[22]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[23]:


df.shape


# In[24]:


df.iloc[488].TotalCharges


# In[25]:


df[df.TotalCharges!=' '].shape


# In[26]:


df1 = df[df.TotalCharges!=' ']
df1.shape


# In[27]:


df1.dtypes


# In[28]:


df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# In[29]:


df1.TotalCharges.values


# In[32]:


df1[df1.Churn=='No']


# In[36]:


tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[38]:


mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges      
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges      

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[39]:


def print_unique_col_values(df):
       for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}')


# In[40]:


print_unique_col_values(df1)


# In[41]:


df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)


# In[42]:


print_unique_col_values(df1)


# In[43]:


yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)


# In[44]:


for col in df1:
    print(f'{col}: {df1[col].unique()}')


# In[45]:


df1['gender'].replace({'Female':1,'Male':0},inplace=True)


# In[46]:


df1.gender.unique()


# In[48]:


# One hot encoding
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
df2.columns


# In[49]:


df2.sample(5)


# In[50]:


df2.dtypes


# In[51]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() #initialising the model
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])


# In[52]:


for col in df2:
    print(f'{col}: {df2[col].unique()}')


# In[53]:


X = df2.drop('Churn',axis='columns')
y = testLabels = df2.Churn.astype(np.float32)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)


# In[54]:


y_train.value_counts()


# In[55]:


y.value_counts()


# In[56]:


5163/1869


# In[57]:


y_test.value_counts()


# In[58]:


X_train.shape


# In[59]:


X_test.shape


# In[60]:


X_train[:10]


# In[62]:


y_test[:10]


# In[63]:


len(X_train.columns)


# In[66]:


get_ipython().system(' pip install tensorflow_addons')


# In[67]:


from tensorflow_addons import losses


# In[68]:


import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix , classification_report


# In[69]:


def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(26, input_dim=26, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    if weights == -1:
        model.fit(X_train, y_train, epochs=100)
    else:
        model.fit(X_train, y_train, epochs=100, class_weight = weights)
    
    print(model.evaluate(X_test, y_test))
    
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)
    
    print("Classification Report: \n", classification_report(y_test, y_preds))
    
    return y_preds


# In[70]:


y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)


# In[71]:


df2.Churn.value_counts()


# In[ ]:




