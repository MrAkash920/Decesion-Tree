#!/usr/bin/env python
# coding: utf-8

# In[7]:


#import library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

#Load dataset
df = pd.read_csv('Loan_Dataset.csv',sep= ',', header= 0)
df.head(10)


# In[8]:


print("Shape of the dataset", df.shape)


# In[11]:


#naming each columns
df.columns = ['Initial payment', 'Last payment', 'Credit Score','House Number', 'Sum', 'Result']
df


# In[12]:


#Information of the dataset
print("Information of the dataset:")
df.info()


# In[19]:


#Seperating target value
X = df.loc[:, ['Initial payment', 'Last payment', 'Credit Score', 'House Number']] 
Y = df.loc[:, ['Result']]


# In[20]:


#splitting dataset
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[21]:


#Function to train dataset
decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
decision_tree.fit(X_train, y_train)


# In[22]:


#Predicting
y_pred = decision_tree.predict(X_test)
y_pred


# In[23]:


#print accuracy score of the predection
print("Acuracy Score: ", accuracy_score (y_test, y_pred))


# In[24]:


#Decesion tree 
text_representation = tree.export_text(decision_tree)
print(text_representation)


# In[29]:


#plot decesion tree
from sklearn.tree import plot_tree
plt.figure(figsize=(25, 20))
plot_tree(decision_tree, feature_names=X.columns, class_names=decision_tree.classes_, filled=True)
plt.show()


# In[ ]:




