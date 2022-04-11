#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[1]:


import numpy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # IMPORTING THE DATASET

# In[2]:


dataset=pd.read_csv(r'E:\ML\New folder\flightdata.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# # ANALYSE THE DATA

# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# In[7]:


dataset.describe().transpose()


# # HANDLING MISSING VALUES

# In[8]:


dataset.isnull().sum()


# In[9]:


dataset.isnull().any()


# In[10]:


dataset['DEST'].unique()


# In[11]:


dataset.drop('Unnamed: 25', axis=1)
dataset.isnull().sum()


# # DATA VISUALIZATION 

# In[12]:


sns.scatterplot(x='ARR_DELAY',y='ARR_DEL15',data=dataset)


# In[13]:


sns.catplot(x='ARR_DEL15',y='ARR_DELAY',kind='bar',data=dataset)


# In[14]:


sns.heatmap(dataset.corr(),cmap='coolwarm',linecolor='white',linewidths=1)


# # FILTER THE DATASET 

# In[15]:


dataset=dataset[["FL_NUM", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_ARR_TIME","DEP_DEL15", "ARR_DEL15"]]


# In[16]:


dataset.isnull().sum()


# In[17]:


dataset[dataset.isnull().any(axis=1)].head()


# In[18]:


dataset['DEP_DEL15'].mode()


# In[19]:


dataset=dataset.fillna({'ARR_DEL15': 1})
dataset=dataset.fillna({'DEP_DEL15': 0})
dataset.iloc[177:185]


# In[20]:


import math

for index, row in dataset.iterrows():
    dataset.loc[index, 'CRS_ARR_TIME'] = math.floor(row['CRS_ARR_TIME'] / 100)


# In[21]:


dataset.head()


# # LABEL ENCODER 

# In[22]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['DEST'] = le.fit_transform(dataset['DEST'])
dataset['ORIGIN'] = le.fit_transform(dataset['ORIGIN'])


# In[23]:


dataset.head()


# In[24]:


dataset['ORIGIN'].unique()


# In[25]:


x=dataset.iloc[:,0:8].values
y=dataset.iloc[:,8:9].values


# In[26]:


x.shape


# In[27]:


y.shape


# # ONE HOT ENCODER

# In[28]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
a=ohe.fit_transform(x[:,4:5]).toarray()
b=ohe.fit_transform(x[:,5:6]).toarray()


# In[29]:


a


# In[30]:


b


# In[31]:


x=np.delete(x,[4,5],axis=1) 


# In[32]:


x.shape


# In[33]:


x=np.concatenate((a,b,x),axis = 1)


# In[34]:


x.shape


# # TRAIN TEST SPLIT

# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[36]:


x_test.shape


# In[37]:


x_train.shape


# In[38]:


y_test.shape


# In[39]:


y_train.shape


# In[40]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# # ALGORITHM

# In[41]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0,)
classifier.fit(x_train,y_train)


# In[42]:


y_pred = classifier.predict(x_test)


# In[43]:


y_pred


# In[44]:


from sklearn.metrics import accuracy_score
desacc = accuracy_score(y_test,y_pred)


# In[45]:


desacc


# In[46]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# In[47]:


cm


# In[48]:


import pickle
pickle.dump(classifier,open('flight.pkl','wb'))

