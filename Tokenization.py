#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
#Natural Language Tool Kit


# In[2]:


nltk.download('punkt')
#Punkt Sentence Tokenizer
#This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model 
#for abbreviation words, collocations, and words that start sentences


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


# In[9]:


#Tokenizing sentences
#return a list of all sentences
sentences = nltk.sent_tokenize(paragraph)
print(sentences)
print(len(sentences))


# In[10]:


#Tokenizing words
words = nltk.word_tokenize(paragraph)
print(words)
print(len(words))

