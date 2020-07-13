#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #for linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for plotting


# In[2]:


trainset = pd.read_csv('train.csv')   #reads csv file into trainset variable
testset = pd.read_csv('test.csv')


# In[3]:


trainset.head(n = 891)  #displays the data, default is 5; n sets the value


# In[4]:


title = trainset["Name"] #specifically CHOOSES a coloumn from the data

train = title.head(n = 9) #specifically DISPLAYS the name coloumn upto 9 entries
train


# In[5]:


y_train = trainset.iloc[:, 1].values
     # data.iloc[n] prints the nth row
     # data.iloc[:,3] prints the nth coloumn with indexing starting from 0 

y_train


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


testset.head() #prints the first five testset data


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# Countplot 
sns.catplot(data = trainset, x = "Sex", hue = "Survived", kind = "count")


# In[8]:


group = trainset.groupby(['Pclass','Survived'])
Pclass_survived = group.size().unstack()

sns.heatmap(Pclass_survived, annot = True, fmt = "d")


# In[9]:


trainset['Family_size'] = 0
trainset['Family_size'] = trainset['Parch'] + trainset['SibSp']

trainset['Alone'] = 0
trainset.loc[trainset.Family_size == 0, 'Alone' ] = 1

sns.factorplot(x = 'Family_size', y = 'Survived', data  = trainset)
sns.factorplot(x = 'Alone', y = 'Survived', data = trainset)


# In[10]:


trainset['Fare_Range'] = pd.qcut(trainset['Fare'], 4) 
sns.barplot(x ='Fare_Range', y ='Survived', data = trainset)


# In[11]:


sns.distplot(trainset['Age'].dropna(), bins=15, kde=True)


# In[12]:


sns.catplot(x ='Embarked', hue ='Survived', kind ='count', col ='Pclass', data = trainset)

sns.catplot(x ='Embarked', hue ='Pclass', kind ='count', data = trainset)


# In[ ]:





# In[13]:


trainset.head()


# In[14]:


extra_eda_cols = ['SibSp', 'Parch', 'Family_size', 'Fare_Range', 'Alone']
trainset = trainset.drop(extra_eda_cols, axis = 1, inplace = False)
trainset.head()


# In[15]:


extra_cols = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']
trainset = trainset.drop(extra_cols, axis = 1, inplace = False)
trainset.head()


# In[16]:


x_train = trainset.drop('Survived', axis = 1, inplace = False)
print(x_train) 


# In[17]:


sns.heatmap(x_train.isnull())


# In[18]:


trainset.isnull().sum()


# In[19]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x_train[['Age']])
x_train[['Age']]= imputer.transform(x_train[['Age']])


#For 'Embarked' column

imputers = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputers.fit(x_train[['Embarked']])
x_train[['Embarked']]= imputers.transform(x_train[['Embarked']])


# In[20]:


x_train.isnull().sum().any()


# In[21]:


x_train.head()


# In[22]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 


#Sex Column  
x_train['Sex']= label_encoder.fit_transform(x_train['Sex']) 

#Embarked Column
x_train['Embarked']= label_encoder.fit_transform(x_train['Embarked'])

x_train.head()


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)


# In[24]:


testset.head()


# In[25]:


testset.isnull().sum().any()


# In[26]:


sns.heatmap(testset.isnull())


# In[27]:


testset.isnull().sum()


# In[28]:


#For 'Age' column

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(testset[['Age']])
testset[['Age']]= imputer.transform(testset[['Age']])


#For 'Embarked' column

imputers = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputers.fit(testset[['Embarked']])
testset[['Embarked']]= imputers.transform(testset[['Embarked']])


# In[29]:


extra_cols_test = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']
testset = testset.drop(extra_cols_test, axis = 1, inplace = False)
testset.head()


# In[30]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 


#Sex Column  
testset['Sex']= label_encoder.fit_transform(testset['Sex']) 

#Embarked Column
testset['Embarked']= label_encoder.fit_transform(testset['Embarked'])


# In[31]:


testset.head()


# In[32]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
test = sc_x.fit_transform(testset)


# In[33]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[34]:


y_pred = classifier.predict(testset)
from sklearn.model_selection import cross_val_score
acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()
acc_Tree


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(testset)
y_pred = classifier.predict(testset)

from sklearn.model_selection import cross_val_score
acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()
acc_Tree


# In[36]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(testset)
y_pred = classifier.predict(testset)

from sklearn.model_selection import cross_val_score
acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()
acc_Tree


# In[37]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[38]:


y_pred = classifier.predict(testset)
from sklearn.model_selection import cross_val_score
acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()
acc_Tree


# In[39]:


from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(testset)
y_pred = classifier.predict(testset)
from sklearn.model_selection import cross_val_score
acc_Tree = cross_val_score(classifier, x_train, y_train, cv=10, scoring='accuracy').mean()
acc_Tree


# In[40]:


accuracy = {'Model' : ['Logistic Regression', 'K- Nearest Neighbor', 'SVC', 'Decision Tree', 'Random Forest'],
                  'Accuracy' : [0.7890, 0.8047, 0.8226, 0.7935, 0.8037]
                 }
all_cross_val_scores = pd.DataFrame(accuracy, columns = ['Model', 'Accuracy'])
all_cross_val_scores.head()


# In[41]:


test_df = pd.read_csv('test.csv')                #initialises the test data
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],       
    'Survived': y_pred                           #survived value calculated by the above machine learning algorithms
})
submission.to_csv('titanic_prediction.csv', index=False)  #data saved in a new file
print('File Saved')


# In[42]:


submission


# In[ ]:





# In[ ]:




