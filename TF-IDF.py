#!/usr/bin/env python
# coding: utf-8

# # TF (Term Frequency)
# * Words are the features and sentences are rows
# * we calculate the ferquency of each word in a sentence
# * formula = (nb of repetition of the word i a sentence/nb of words in the sentence)

# # IDF (Inverse document frequency
# * formula for each word = log(Nb of sentences/nb of sentences containig that word)

# # Finally
# * we calculate this product (TF * IDF) for each cell

# In[1]:


import nltk


# In[2]:


paragraph = '''First of all love and friendship is relationship, which requires time, effort and many other characteristics,
which form mutual relationship. Both, love and friendship assume emotional involvement, care, respect and devotion. 
Both feelings undergo transformations with the flow of time and go through different stages. 
You can have several friends but situations of being in loving relationships with several people are very rare and are 
regarded rather like deviation, than something normal. This is one of the differences between love and friendship. 
Another incontestable difference between love and friendship is sexual attraction. 
Friendship usually doesn’t assume any sexual attraction at all and most forms of love (in the context of relationship 
between people who are in love) include sexual attraction. Love can be reciprocal or not, and in this field it’s expressed 
more like an attitude. It’s hard to imagine reciprocal friendship. It either exists or not. Friendship is rather a 
relationship, which emerges on the joint of several spheres i.e. emotional, mental, physical, etc.'''


# In[3]:


# Cleaning the text
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[4]:


wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i]) 
    #this line is replacing every word, that's not starting by [a-zA-Z], by ' '
    review = review.lower()
    review = review.split()
    #review now is a list of words for each sentences[i]
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# In[5]:


# Creating TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)


# In[ ]:




