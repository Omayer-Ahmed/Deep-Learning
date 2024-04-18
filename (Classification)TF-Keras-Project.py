#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('onlinefoods.csv',sep=',')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df['Output'].value_counts()


# In[7]:


df['Output'].value_counts().plot(kind='bar')


# In[8]:


df['Feedback'].value_counts()


# In[9]:


df['Feedback'].value_counts().plot(kind='bar')


# In[10]:


df['Family size'].value_counts().plot(kind='bar')


# In[11]:


import pandas as pd

columns_to_encode = ['Gender', 'Marital Status','Occupation','Monthly Income','Educational Qualifications'
                     ,'Feedback']
dummies = pd.get_dummies(df[columns_to_encode]).astype(int)


# In[12]:


dummies.head()


# In[13]:


merge=pd.concat([df,dummies],axis='columns')
merge.head()


# In[14]:


newdf=merge.drop(['Gender','Marital Status','Occupation','Monthly Income',
               'Educational Qualifications','Feedback'],axis=1)


# In[15]:


newdf.tail()


# In[16]:


mapping={'Yes':1,'No':0}
newdf['Output']=newdf['Output'].replace(mapping)


# In[17]:


newdf.head()


# In[18]:


newdf.shape


# In[19]:


x=newdf.drop(['Output'],axis=1)


# In[20]:


x.head()


# In[21]:


y=newdf.iloc[:,5:6]


# In[22]:


y.head()


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20,random_state=43)


# In[25]:


xtrain.shape


# In[26]:


xtest.shape


# In[27]:


ytrain.shape


# In[28]:


ytest.shape


# In[29]:


from sklearn.preprocessing import MinMaxScaler


# In[30]:


scaler=MinMaxScaler()


# In[31]:


xtrain=scaler.fit_transform(xtrain)


# In[32]:


ytest=scaler.transform(xtest)


# In[33]:


xtrain.shape


# In[34]:


xtest.shape


# In[35]:


ytest.shape


# In[36]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[37]:


model=Sequential()


# In[38]:


model.add(Dense(26,input_dim=26,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# Common loss functions include mean squared error (MSE), categorical cross-entropy, and binary cross-entropy, depending on the type of problem being 

# In[39]:


#compiling model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[40]:


xtrain.shape


# In[41]:


ytrain.shape


# In[42]:


#fitting model
model.fit(xtrain,ytrain,epochs=150,batch_size=10)


# In[43]:


loss_df=pd.DataFrame(model.history.history)
loss_df


# In[44]:


loss_df.plot()


# In[45]:


#train accuracy
model.evaluate(xtrain,ytrain)


# In[46]:


xtest.shape


# In[47]:


ytest.shape


# In[52]:


#testing accuracy
model.evaluate(xtest,ytest)


# In[53]:


loss,accuracy=model.evaluate(xtest,ytest,batch_size=50)


# In[54]:


# Assuming model is your trained neural network model
ypred_prob = model.predict(xtest)
y_pred = np.argmax(ypred_prob, axis=1)


# In[55]:


from sklearn.metrics import mean_squared_error

# Assuming y_test and y_pred are your target variable and predicted values, respectively.
mse = mean_squared_error(ytest, y_pred)
print("Mean Squared Error:", mse)


# In[ ]:




