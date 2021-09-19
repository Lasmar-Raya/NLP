#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


path = "D:/NLP by krish/spam_classifier/SMSSpamCollection"


# In[3]:


messages = pd.read_csv(path, sep='\t', names=["label", "message"])


# In[4]:


messages.info()


# In[5]:


messages.describe()


# In[6]:


messages.head()


# In[7]:


print(len(messages))


# In[8]:


words = WordNetLemmatizer()
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) 
    #this line is replacing every word, that's not starting by [a-zA-Z], by ' '
    review = review.lower()
    review = review.split()
    #review now is a list of words for each sentences[i]
    review = [words.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[9]:


# Creating BOW model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)


# In[10]:


X.shape


# In[11]:


# Creating a TF-IDF model and Reduce features
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
print(X)


# In[12]:


X.shape


# In[13]:


y = messages['label']
print (y)


# In[14]:


# Getting label
y = pd.get_dummies(messages['label']) 
# 'ham' and 'spam' are categorical variables we can use get_dummies, it returns a dataframe that contains 2 columns labeled ham and spam, if the message is a spam it will get value 1 in the comumn spam else 0 and vice versa 
y = y.iloc[:,1].values
# since we have only 2 variables, we can reduce the number of columns to 1


# In[15]:


#Splitting the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[16]:


# Train using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)


# In[17]:


print(y_pred)


# In[32]:


# Compare results with y_test

from sklearn.metrics import confusion_matrix
confusion_m = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=[0,1]))


# In[33]:


print(confusion_m)
#rows actual output
#columns predicted output
# 955+134 are correctly predicted


# In[35]:


# Check the accuracy

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


# In[ ]:




