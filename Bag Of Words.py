#!/usr/bin/env python
# coding: utf-8

# # Bag of words
# * After lowering sentences by stopwords we're going to make a histogram to show the frequency of each word but first, we need to sort them (asc=false)
# * Converting words to vectors is called Binary BOW
# * To do that we're going to consider all selected words as features and for each sentence if a certain word exists, it will take 1 else 0 and hence, each sentence is represented by a vector
# * If a certain word is repeated in the same sentence we can just increment the counter
# * The problem here is that words that are represented in a sentence, they have the same representation so semantically we cannot tell which word is more important 

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


# In[8]:


# Cleaning the text
import re #regular expression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[9]:


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


# In[10]:


print(sentences)


# In[11]:


print(corpus)


# In[15]:


# Creating the Bag Of Words model
#Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)


# In[16]:


X.shape
# X is the representation that we explained in the description above
# 59 = nb of features(words)
# 12 nb of sentences

