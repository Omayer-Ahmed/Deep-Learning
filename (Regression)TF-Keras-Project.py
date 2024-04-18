#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('Student_Performance.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


columns_encoded=['Extracurricular Activities']
dummies=pd.get_dummies(df[columns_encoded]).astype(int)


# In[9]:


dummies.head()


# In[10]:


merge=pd.concat([df,dummies],axis='columns')


# In[11]:


merge.head()


# In[12]:


newdf=merge.drop(['Extracurricular Activities'],axis=1)


# In[13]:


newdf.head()


# In[14]:


newdf.corr()


# In[15]:


sb.heatmap(newdf.corr())


# In[16]:


x=newdf.drop(['Performance Index'],axis=1)


# In[17]:


x.head()


# In[18]:


y=newdf.iloc[:,4:5]


# In[19]:


y.head()


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20,random_state=42)


# In[22]:


xtrain.shape


# In[23]:


ytrain.shape


# In[24]:


xtest.shape


# In[25]:


ytest.shape


# In[26]:


from sklearn.preprocessing import MinMaxScaler


# In[27]:


scaler=MinMaxScaler()


# In[28]:


xtrain=scaler.fit_transform(xtrain)


# In[29]:


ytest=scaler.transform(xtest)


# In[30]:


xtrain.shape


# In[31]:


ytest.shape


# In[32]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[33]:


model=Sequential()


# In[34]:


model.add(Dense(6,input_dim=6,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1,activation='relu'))


# In[35]:


#compiling model
model.compile(loss='mse',optimizer='adam')


# In[36]:


#fitting model
model.fit(x=xtrain,y=ytrain,validation_data=(xtest,ytest),epochs=150,batch_size=64)


# In[37]:


loss_df=pd.DataFrame(model.history.history)
loss_df


# In[38]:


loss_df.plot()


# In[ ]:





# In[ ]:




