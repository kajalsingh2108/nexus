#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # vizulization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re


# ## Loading data

# In[3]:


df=pd.read_csv("Twitter.csv",encoding = 'latin',header=None)


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df=df.rename(columns={0: 'sentiment',1:"id",2:"Date",3:"flag",4:"user",5:"text"})
df.sample(5)


# In[7]:


df = df.drop(['id', 'Date', 'flag', 'user'], axis=1)


# In[8]:


lab_to_sentiment = {0:"Negative", 4:"Positive"}
def label_decoder(label):
  return lab_to_sentiment[label]
df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))


# In[9]:


df["text"][0]


# In[10]:


df.info()


# In[11]:


df.duplicated().sum()


# In[12]:


df = df.sample(50000)


# In[13]:


df


# In[14]:


def remove_abb(data):
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"there's", "there is", data)
    data = re.sub(r"We're", "We are", data)
    data = re.sub(r"That's", "That is", data)
    data = re.sub(r"won't", "will not", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"Can't", "Cannot", data)
    data = re.sub(r"wasn't", "was not", data)
    data = re.sub(r"don\x89Ûªt", "do not", data)
    data= re.sub(r"aren't", "are not", data)
    data = re.sub(r"isn't", "is not", data)
    data = re.sub(r"What's", "What is", data)
    data = re.sub(r"haven't", "have not", data)
    data = re.sub(r"hasn't", "has not", data)
    data = re.sub(r"There's", "There is", data)
    data = re.sub(r"He's", "He is", data)
    data = re.sub(r"It's", "It is", data)
    data = re.sub(r"You're", "You are", data)
    data = re.sub(r"I'M", "I am", data)
    data = re.sub(r"shouldn't", "should not", data)
    data = re.sub(r"wouldn't", "would not", data)
    data = re.sub(r"i'm", "I am", data)
    data = re.sub(r"I\x89Ûªm", "I am", data)
    data = re.sub(r"I'm", "I am", data)
    data = re.sub(r"Isn't", "is not", data)
    data = re.sub(r"Here's", "Here is", data)
    data = re.sub(r"you've", "you have", data)
    data = re.sub(r"you\x89Ûªve", "you have", data)
    data = re.sub(r"we're", "we are", data)
    data = re.sub(r"what's", "what is", data)
    data = re.sub(r"couldn't", "could not", data)
    data = re.sub(r"we've", "we have", data)
    data = re.sub(r"it\x89Ûªs", "it is", data)
    data = re.sub(r"doesn\x89Ûªt", "does not", data)
    data = re.sub(r"It\x89Ûªs", "It is", data)
    data = re.sub(r"Here\x89Ûªs", "Here is", data)
    data = re.sub(r"who's", "who is", data)
    data = re.sub(r"I\x89Ûªve", "I have", data)
    data = re.sub(r"y'all", "you all", data)
    data = re.sub(r"can\x89Ûªt", "cannot", data)
    data = re.sub(r"would've", "would have", data)
    data = re.sub(r"it'll", "it will", data)
    data = re.sub(r"we'll", "we will", data)
    data = re.sub(r"wouldn\x89Ûªt", "would not", data)
    data = re.sub(r"We've", "We have", data)
    data = re.sub(r"he'll", "he will", data)
    data = re.sub(r"Y'all", "You all", data)
    data = re.sub(r"Weren't", "Were not", data)
    data = re.sub(r"Didn't", "Did not", data)
    data = re.sub(r"they'll", "they will", data)
    data = re.sub(r"they'd", "they would", data)
    data = re.sub(r"DON'T", "DO NOT", data)
    data = re.sub(r"That\x89Ûªs", "That is", data)
    data = re.sub(r"they've", "they have", data)
    data = re.sub(r"i'd", "I would", data)
    data = re.sub(r"should've", "should have", data)
    data = re.sub(r"You\x89Ûªre", "You are", data)
    data = re.sub(r"where's", "where is", data)
    data = re.sub(r"Don\x89Ûªt", "Do not", data)
    data = re.sub(r"we'd", "we would", data)
    data = re.sub(r"i'll", "I will", data)
    data = re.sub(r"weren't", "were not", data)
    data = re.sub(r"They're", "They are", data)
    data = re.sub(r"Can\x89Ûªt", "Cannot", data)
    data = re.sub(r"you\x89Ûªll", "you will", data)
    data = re.sub(r"I\x89Ûªd", "I would", data)
    data = re.sub(r"let's", "let us", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"can't", "cannot", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"i've", "I have", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"i'll", "I will", data)
    data = re.sub(r"doesn't", "does not",data)
    data = re.sub(r"i'd", "I would", data)
    data = re.sub(r"didn't", "did not", data)
    data = re.sub(r"ain't", "am not", data)
    data = re.sub(r"you'll", "you will", data)
    data = re.sub(r"I've", "I have", data)
    data = re.sub(r"Don't", "do not", data)
    data = re.sub(r"I'll", "I will", data)
    data = re.sub(r"I'd", "I would", data)
    data = re.sub(r"Let's", "Let us", data)
    data = re.sub(r"you'd", "You would", data)
    data = re.sub(r"It's", "It is", data)
    data = re.sub(r"Ain't", "am not", data)
    data = re.sub(r"Haven't", "Have not", data)
    data = re.sub(r"Could've", "Could have", data)
    data = re.sub(r"youve", "you have", data)  
    data = re.sub(r"donå«t", "do not", data)
    
    return data


# In[15]:


df['text'] = df['text'].apply(remove_abb)
df["text"]


# In[18]:


get_ipython().system('pip install textblob')
get_ipython().system('pip install nltk')


# In[19]:


import nltk

# Download the Punkt tokenizer
nltk.download('punkt')


# In[20]:


from nltk.tokenize import word_tokenize


# In[21]:


df['tokenized_text'] = df['text'].apply(word_tokenize)


# In[22]:


df.head()


# In[23]:


import nltk

# Download the stopwords resource
nltk.download('stopwords')


# In[24]:


from nltk.corpus import stopwords
import nltk

# Download the stopwords data if not already downloaded
nltk.download('stopwords')

stopwords.words("english")


# In[25]:


def remove_stopwords(text):
    
    L = []
    for word in text:
        if word not in stopwords.words('english'):
            L.append(word)
            
    return L


# In[ ]:


df['tokenized_text'] = df['tokenized_text'].apply(remove_stopwords)


# In[ ]:


df.head()


# In[ ]:


df['text'] = df['tokenized_text'].apply(lambda x:" ".join(x))
df.head()


# In[37]:


df['char_length'] = df['text'].str.len()
df.head()


# In[32]:


df['word_length'] = df['tokenized_text'].apply(len)



# In[33]:


df.head()


# ## Data Visualization

# In[34]:


sns.displot(df["word_length"])


# In[35]:



sns.distplot(df[df['sentiment'] == 'Positive']['word_length'])
sns.distplot(df[df['sentiment'] == 'Negative']['word_length'])
sns.plot


# In[38]:


sns.distplot(df[df['sentiment'] == 'Positive']['char_length'])
sns.distplot(df[df['sentiment'] == 'Negative']['char_length'])


# In[39]:


get_ipython().system('pip install wordcloud')


# In[40]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.figure(figsize = (20,20)) # Positive Review Text
wc = WordCloud(width = 1000 , height = 400).generate(" ".join(df[df['sentiment'] == 'Negative']['text']))
plt.imshow(wc)


# In[46]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.figure(figsize = (20,20)) # Positive Review Text
wc = WordCloud(width = 959 , height = 430).generate(" ".join(df[df['sentiment'] == 'Positive']['text']))
plt.imshow(wc)


# In[42]:


# Count the occurrences of each sentiment
sentiment_counts = df['sentiment'].value_counts()


# In[45]:


#Create a bar chart
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['darkblue', 'green'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()


# ##Twitter Sentiment Analysis Project Documentation
# 
# #Project Overview:
# The Twitter Sentiment Analysis project involves analyzing a dataset of tweets to determine the sentiment expressed in each tweet, categorizing it as either positive or negative. The goal is to gain insights into public opinions and sentiments shared on Twitter, utilizing various data preprocessing and visualization techniques.
# 
# #Code Implementation:
# The provided Python code demonstrates the data preprocessing steps and sentiment visualization using various libraries such as pandas, matplotlib, seaborn, and wordcloud. Here's a brief overview of the code:
# 
# #Data Loading and Exploration:
# 
# Reads the dataset ('tweets.csv') using pandas.
# Renames columns for better readability.
# Drops unnecessary columns ('id', 'Date', 'flag', 'user').
# Data Cleaning:
# 
# Decodes sentiment labels (0 and 4) into meaningful categories (Negative and Positive).
# Handles duplicate entries and missing values.
# Reduces the dataset size to 50,000 randomly sampled tweets.
# Text Preprocessing:
# 
# Converts text to lowercase.
# Removes leading and trailing whitespaces.
# Removes HTML tags and URLs.
# Expands abbreviations.
# Removes punctuation.
# Tokenizes the text.
# Removes stop words.
# Exploratory Data Analysis (EDA):
# 
# Visualizes the distribution of sentiment labels using a bar chart.
# Analyzes word and character lengths distribution.
# Word Clouds:
# 
# Generates word clouds for both positive and negative sentiments to visualize frequent words.
# Dependencies:
# 
# The code requires the installation of various Python libraries, including pandas, matplotlib, seaborn, wordcloud, and nltk. Ensure they are installed before running the code.
# Conclusion:
# The Twitter Sentiment Analysis project aims to provide a preliminary understanding of sentiment distribution in a Twitter dataset. The documentation serves as a guide for understanding the project objectives, steps, and code implementation. Further development may include sentiment prediction models and more in-depth analysis of feature importance.

# In[ ]:




