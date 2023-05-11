#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


d_frame = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
d_frame


# In[21]:


print(d_frame.shape)


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(d_frame.Hours,d_frame.Scores)
plt.title("Hours Vs Score")
plt.xlabel("hours")
plt.ylabel("scores")


# In[29]:


d_frame.info()


# In[30]:


d_frame.corr()


# In[33]:


reg = linear_model.LinearRegression()
reg.fit(d_frame[['Hours']].values,d_frame.Scores)


# In[34]:


reg.predict([[9.5]])


# In[45]:


plt.title("Hours Vs Score")
plt.xlabel("hours")
plt.ylabel("scores")
plt.scatter(d_frame.Hours,d_frame.Scores)
plt.plot(d_frame.Hours,reg.predict(d_frame[['Hours']].values),color= "blue")


# In[46]:


#error checking
from sklearn import metrics
print(mean_absolute_error(predict(9.5)))


# In[ ]:




