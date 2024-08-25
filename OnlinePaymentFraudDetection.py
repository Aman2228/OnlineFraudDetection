
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[4]:


df = pd.read_csv('onlinefraud.csv')


# In[5]:


df = df.iloc[:,0:10]


# In[6]:


df


# In[7]:


df = df.drop(columns=['nameDest','nameOrig'])


# In[8]:


df


# In[3]:


clf = DecisionTreeClassifier()


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(df.drop('isFraud',axis=1),df['isFraud'],test_size=0.3,random_state=0)


# In[11]:


x_train


# In[12]:


df.isnull().sum()


# In[13]:


df.info()


# In[19]:


#OneHotEncoding on Type
trf = ColumnTransformer([
    ('encode type',OneHotEncoder(sparse=False),[1])
],remainder = 'passthrough')


# In[20]:


trf


# In[21]:


x_train_transformed = trf.fit_transform(x_train)


# In[22]:


x_test_transformed = trf.transform(x_test)


# In[23]:


y_train


# In[24]:


clf.fit(x_train_transformed,y_train.values)


# In[25]:


y_pred = clf.predict(x_test_transformed)


# In[26]:


accuracy_score(y_pred,y_test.values)


# In[28]:


f1_score(y_pred,y_test.values)

