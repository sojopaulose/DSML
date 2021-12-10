#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('salary.csv')


# In[41]:


X = dataset.iloc[:,:-1]
X


# In[46]:


X = dataset.iloc[:,:-1]
X


# In[47]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 1)


# In[48]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[49]:


y_pred = regressor.predict(X_test)
y_pred


# In[50]:


y_test


# In[51]:


plt.scatter(X_train,y_train, color='red',)
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()


# In[52]:


plt.scatter(X_test,y_test, color='red',)
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:




