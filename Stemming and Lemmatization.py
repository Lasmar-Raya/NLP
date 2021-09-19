#!/usr/bin/env python
# coding: utf-8

# ### Stemming & Lemmatization
# Process of reducing infected words to their word stem(bass word)
# ### Stemming
# #### In general, with some of the words, stemming may not give a meaningful representation  
# finally, final, finalized --> fina
# going, goes, gone --> go
# history, historical --> histori
# ### Lemmatization
# #### It gives an understandable presentation
# finally, final, finalized --> final
# going, goes, gone --> go
# history, historical --> history

# # Stemming

# In[2]:


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#Stopwords are the English words which does not add much meaning to a sentence.


# In[3]:


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


# In[4]:


sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()
# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words
                if word not in set(stopwords.words('english'))]
    sentences[i]= ' '.join(words)


# In[6]:


print(sentences)


# In[7]:


#Sets are a mutable collection of distinct (unique) immutable values that are unordered.
stopwords.words('english')


# In[9]:


x = set(['python', 'R', 'java'])
print(x)
# notice that the values in the set are not in the order added in. 
#This is because sets are unordered


# # Lemmatization

# In[14]:


import nltk
nltk.download()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[11]:


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


# In[15]:


sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()
# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words
                if word not in set(stopwords.words('english'))]
    sentences[i]= ' '.join(words)


# In[16]:


print(sentences)

