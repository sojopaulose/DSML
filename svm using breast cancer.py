#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


X =cancer.data
y = cancer.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[5]:


from sklearn.svm import SVC


# In[12]:


model = SVC()
model.fit(X_train,y_train)


# In[14]:


predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


# In[ ]:





# In[ ]:




