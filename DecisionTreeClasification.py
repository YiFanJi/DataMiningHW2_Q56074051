#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Generate the feature data and answer

# In[205]:


data=['Light','Space','Rent','Electricity','Rank']
# data[1:500] = [np.zeros((500,4))]
a = np.zeros((499,5))
# print(len(a))
#data.append(a)

for i in range(0,499):
    a[i]=[random.randrange(1,3,1),random.randrange(10,25,1),random.randrange(2000,10000,1),random.randrange(3,5,1),0]
# print(a)  
for i in range(0,499):
    if a[i][0]>=2 and a[i][1]>15 and a[i][2]<5500 and a[i][3]==3:
        a[i][4]=1
    
    elif a[i][0]==1 or a[i][1]<10 and a[i][2]>8000 or a[i][3]==5:
        a[i][4]=3
        
    else:
        a[i][4]=2
    
df = pd.DataFrame(a.T, index=data)
df = df.T
df


# In[207]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df[['Light','Space','Rent','Electricity']],df[['Rank']],test_size=0.3,random_state=0)


# In[210]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)
tree.fit(X_train,Y_train)


# In[211]:


tree.score(X_test,Y_test['Rank'])


# In[ ]:




