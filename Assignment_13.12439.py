
# coding: utf-8

# # Problem Statement:

# ## In this assignment I will build the random forest model after normalizing the variable to house pricing from boston data set.

# ## Importing Modules

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Pre-Processing

# In[24]:


boston = datasets.load_boston() #Load boston dataset from datasets
features = pd.DataFrame(boston.data, columns=boston.feature_names)  #Reading data into a DataFrame
targets = boston.target 


# In[25]:


#The boston variable is a dictionary and its key values are as follows: 
print(boston.keys())


# In[26]:


features.head() #Returns the first 5 rows of features dataframe


# In[27]:


targets #Show the targets values


# ## Data Exploration/Analysis

# In[28]:


#Description of the boston data
print(boston.DESCR)


# In[29]:


features.shape #Shape of the features


# In[30]:


targets.shape #Shape of the targets


# In[31]:


features.info() #Prints information about features DataFrame.


# In[32]:


features.describe() #The summary statistics of the dataframe


# In[33]:


features.isnull().values.any() #Check for any NA’s in the dataframe.


# ## Data Visualization

# In[34]:


sns.pairplot(features) #Plots pairwise relationships in a dataset.


# ## Train, Test & Split

# In[35]:


#Splitting the datasets into train and test datasets (80% training dataset and 20% test dataset)
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=42)

#Noramalizing training and test datasets
scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)


# ## Creating and Training the Model

# In[36]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, y_train)


# ## Predicting Price using Test Data

# In[37]:


y_pred = rf.predict(X_test) #Calculating the prediction values


# In[38]:


y_pred.shape #Prediction shape from test data


# In[39]:


#To visualize the differences between actual prices and predicted values, creating a scatter plot.
sns.set_style("whitegrid")
sns.set_context("poster")
plt.figure(figsize=(16,9))
plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.text(15,-5, "Comparison between the actual prices and predicted prices.", ha='left')
plt.show()


# In[40]:


sns.regplot(y_test, y_pred, data=features, fit_reg=True) #Plot y_test and y_pred 


# In[41]:


sns.regplot(x=rf.predict(features), y=targets, data=features, fit_reg=True) #Plot predicted and actual Price values.


# ## Evaluating the Model

# In[42]:


from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
predicted_train = rf.predict(X_train)
predicted_test = rf.predict(X_test)
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)
print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
print(f'Test data R-2 score: {test_score:>5.3}')
print(f'Test data Spearman correlation: {spearman[0]:.3}')
print(f'Test data Pearson correlation: {pearson[0]:.3}')


# In[43]:


#Calculating Mean Squared Error
mse = sklearn.metrics.mean_squared_error(y_test, y_pred) #Mean Squared Error: To check the level of error of a model
print(mse)


# In[44]:


#Calculating Root Mean Squared Error#Calcula 
rmse = mse ** 0.5 #Square root of mse (Mean Squared Error)
print("The Root Mean Square Error (RMSE) for the Model is "+ str(rmse) +" and the Results can be further improved using feature extraction and rebuilding, training the model.")

