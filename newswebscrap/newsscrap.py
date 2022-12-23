import requests 
import nltk 
from bs4 import BeautifulSoup 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import random 
from wordcloud import wordcloud 
import os 
import spacy 
nlp=spacy.load('en_core_web_sm')
from textblob import TextBlob 
from pattern.en import sentiment 

r = requests.get('https://www.timesnownews.com/')
r.encoding='utf-8'
html=r.text 
soup=BeautifulSoup(html)
text=soup.get_text() 

#cleaning the data
clean_text= text.replace("n", " ")
clean_text= clean_text.replace("/", " ")       
clean_text= ''.join([c for c in clean_text if c != "'"]) 

#format to sentences
sentence=[]
tokens=nlp(clean_text)
for sent in tokens.sents:
    sentence.append(sent.text.strip())

#sentiment analysis 
sentiment=[]
for s in sentence:
    txt=TextBlob(s) 
    a= txt.sentiment.polarity
    b= txt.sentiment.subjectivity
    sentiment.append([s,a,b]) 

#making it as a dataframe 
dataframe = pd.DataFrame(sentiment, columns=[
                           'Sentence', 'Polarity', 'Subjectivity']) 
print(dataframe)
dataframe.head()
dataframe.info() 


sns.displot(dataframe["Sentence"], height=5, aspect=1.8)
plt.xlabel("Polarity")

sns.displot(dataframe["Polarity"], height=5, aspect=1.8)
plt.xlabel("Polarity")

sns.displot(dataframe["Subjectivity"], height=5, aspect=1.8)
plt.xlabel("Subjectvity")
