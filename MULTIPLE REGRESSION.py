#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[10]:


dataset = pd.read_csv('startups.csv')

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]
x


# In[11]:


states=pd.get_dummies(x['State'], drop_first=True)


# In[12]:


x = x.drop('State',axis=1)


# In[13]:


x = pd.concat([x,states],axis=1)


# In[14]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[15]:



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[17]:



y_pred = regressor.predict(x_test)


# In[20]:


from sklearn.metrics import r2_score

score=r2_score(y_test,y_pred)
print(f'R2 score: {score}')


# In[19]:


plt.figure(figsize = (5, 5))
plt.scatter(y_test, y_pred)
plt.title('Actual vs Prdicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[ ]:




