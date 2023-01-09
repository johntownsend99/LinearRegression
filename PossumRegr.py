#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# Using the [Possum Regression Dataset](https://www.kaggle.com/datasets/abrambeyer/openintro-possum), create a linear regression model to predict a possum's head length based on its total body length.

# In[18]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# In[19]:


# read in data and perform EDA
possums = pd.read_csv('possum.csv')

possums.info()

print("\n first 5 records:\n", possums.head())

print("\n shape:\n", possums.shape)


# ### Total length to predict a possum's head length

# In[20]:


# select columns
possum_df = possums[['totlngth','hdlngth']]
possum_df.head()


# In[21]:


# display scatterplot with regression line to show correlation between two variables
sns.lmplot(x='totlngth',y='hdlngth',data=possum_df)


# In[22]:


# drop any row with null values 
# NOTE: it's not always the best method to drop a record where any column contains a null value, but since there is only one
#independent and one dependent variable, the model's accuracy could be affected by any number of null values
print("initial df length:\n", possum_df.count())
possum_df=possum_df.dropna()
print("df length after removing null vals:\n", possum_df.count())


# In[23]:


# select independent (X) and dependent (y) variable
X = np.array(possum_df['totlngth']).reshape(-1,1)
y = np.array(possum_df['hdlngth']).reshape(-1,1)


# In[24]:


# split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Why do we need to split our data into testing and training datasets?
# 
# This [video](https://learn.microsoft.com/en-us/shows/dev-intro-to-data-science/why-do-you-split-data-into-testing-and-training-data-in-data-science-12-of-28) explains it. 

# In[25]:


# initialize LinReg object
regr = LinearRegression()

# fit model 
regr.fit(X_train, y_train)

# return coefficient of determination (R^2)
print(round(regr.score(X_test,y_test), 4))


# In[26]:


# predict head length
y_pred = regr.predict(X_test)

#explore results
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[27]:


# evaluate model with mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(mae)


# ## Challenge
# 
# Use age, sex, and totlength to predict a possum's trapping location (site). Using linear regress to predict a target variable (y) with a set of variables (X1, X2...Xn) is called a multivariate regression. 

# In[142]:


# determine which columns are body dimensions
possums.head()


# In[143]:


df = possums[['sex','age','totlngth','hdlngth']].copy()
df = df.dropna()


# In[144]:


X = df[['sex','age','totlngth']]
y = df['hdlngth']


# In[147]:


# convert sex into binary values
le = preprocessing.LabelEncoder()
X['sex']=le.fit_transform(X['sex'])


# In[148]:


# scale age and totlngth to be in same data range
scaler = preprocessing.MinMaxScaler()
X[['age','totlngth']]=scaler.fit_transform(X[['age','totlngth']])


# In[150]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[152]:


# initialize LinReg object
regr = LinearRegression()

# fit model 
regr.fit(X_train, y_train)

# return coefficient of determination (R^2)
print(round(regr.score(X_test,y_test), 4))


# In[159]:


# predict head length
y_pred = regr.predict(X_test)


# In[161]:


# evaluate model accuracy
mae = mean_absolute_error(y_test, y_pred)
print(mae)


# In[165]:


# visually compare predictions to y_test
prediction = pd.Series(y_pred)
true = y_test.reset_index(drop=True)
z = pd.concat([true,prediction],axis=1)
z.columns = ['True','Prediction']
z


# In[ ]:




