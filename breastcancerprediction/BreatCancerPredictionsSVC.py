#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[5]:


cancer = load_breast_cancer()


# In[6]:


cancer


# In[7]:


cancer.keys()


# In[8]:


print(cancer['DESCR'])


# In[9]:


print(cancer['feature_names'])


# In[10]:


cancer['data'].shape


# In[12]:


df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(cancer['feature_names'],['target']))


# In[16]:


sns.pairplot(df_cancer, hue='target', vars=['mean radius' ,'mean texture', 'mean perimeter' ,'mean area',
 'mean smoothness'])


# In[17]:


sns.countplot(df_cancer['target'])


# In[18]:


sns.scatterplot(x='mean area' , y = 'mean smoothness', hue='target', data = df_cancer)


# In[19]:


plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(),annot = True)


# In[21]:


X = df_cancer.drop(['target'], axis = 1)


# In[24]:


X


# In[25]:


Y = df_cancer['target']


# In[26]:


Y


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)


# In[30]:


X_test


# In[31]:


from sklearn.svm import SVC


# In[33]:


from sklearn.metrics import classification_report, confusion_matrix


# In[35]:


svc_model = SVC()


# In[36]:


svc_model.fit(X_train,y_train)


# In[37]:


y_predict = svc_model.predict(X_test)


# In[38]:


y_predict


# In[39]:


cm = confusion_matrix(y_test,y_predict)


# In[40]:


sns.heatmap(cm, annot=True)


# In[41]:


min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train-min_train)/range_train


# In[42]:


sns.scatterplot(x= X_train_scaled['mean area'], y= X_train_scaled['mean smoothness'], hue=y_train)


# In[43]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test-min_test)/range_test


# In[44]:


svc_model.fit(X_train_scaled, y_train)


# In[45]:


y_predict = svc_model.predict(X_test_scaled)


# In[46]:


cm = confusion_matrix(y_test,y_predict)


# In[47]:


sns.heatmap(cm, annot=True)


# In[48]:


print(classification_report(y_test,y_predict))


# In[49]:


param_grid = { 'C' : [0.1,1,10,100] , 'gamma':[1,0.1,0.01,0.001] , 'kernel' : ['rbf']}


# In[50]:


from sklearn.model_selection import GridSearchCV


# In[51]:


grid = GridSearchCV(SVC(), param_grid , refit=True , verbose = 4 )


# In[53]:


grid.fit(X_train_scaled,y_train)


# In[54]:


grid.best_params_


# In[55]:


grid_predictions = grid.predict(X_test_scaled)


# In[56]:


cm = confusion_matrix(y_test, grid_predictions)


# In[57]:


sns.heatmap(cm,annot=True)


# In[60]:


print(classification_report(y_test,grid_predictions))


# In[ ]:




