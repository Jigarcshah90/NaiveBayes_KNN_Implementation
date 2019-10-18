#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Implementation of KNN, Naive Bayes, Decision Tree. 
## in order to know the person who is going to be defaulted in payment next month.

#Importing librairies

import pandas as pd 
import numpy as np

# Scikit-learn library: For SVM
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm

import itertools

# Matplotlib library to plot the charts
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Library for the statistic data vizualisation
import seaborn

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('UCI_Credit_Card.csv') # Reading the file .csv
df = pd.DataFrame(data) # Converting data to Panda DataFrame


# In[5]:


data.head()


# In[22]:


##Splitting the data into dependent and independent variable

X = data.drop(['default.payment.next.month'], axis=1)
Y = data['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[23]:


#Decision Tree

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[24]:


### Implementation of decision tree using criterion as Gini###
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)
accuracy_score(y_test,y_pred_gini)*100


# In[11]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


# In[25]:


y_pred_en = clf_entropy.predict(X_test)


# In[26]:


accuracy_score(y_test,y_pred_en)*100


# In[27]:


#implementation of KNN model

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=1)  
classifier.fit(X_train, y_train)
y_pred_knn = classifier.predict(X_test)


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred_knn))  
print(classification_report(y_test, y_pred_knn)) 


# In[30]:


accuracy_score(y_test,y_pred_knn)*100


# In[33]:


## Very basic Naive Bayes implementation

from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(X_train, y_train)

y_pred_nb = model.predict(X_test)

accuracy_score(y_test,y_pred_nb)*100


# So using the decision tree we get the highest accuracy of 81.83 % to predict the person who will be defaulted for the payment in next month. 

# In[ ]:




