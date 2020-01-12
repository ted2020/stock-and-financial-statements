#!/usr/bin/env python
# coding: utf-8

# In[2]:


import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[3]:


amazon = quandl.get("WIKI/AMZN")
print(amazon.head())


# In[4]:


amazon = amazon[['Adj. Close']]
print(amazon.head())


# In[5]:


forecast_len=30
amazon['Predicted'] = amazon[['Adj. Close']].shift(-forecast_len)
print(amazon.tail())


# In[6]:


x=np.array(amazon.drop(['Predicted'],1))
x=x[:-forecast_len]
print(x)


# In[7]:


y=np.array(amazon['Predicted'])
y=y[:-forecast_len]
print(y)


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[9]:


svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
svr_rbf.fit(x_train,y_train)


# In[10]:


svr_rbf_confidence=svr_rbf.score(x_test,y_test)
print(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")


# In[11]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[12]:


lr_confidence=lr.score(x_test,y_test)
print(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




