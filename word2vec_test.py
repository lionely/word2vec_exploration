#Importing the libraries
import numpy as np
import re
import time 
from gensim.models import Word2Vec


############### Part 1 - Data Preprocessing ###############

# Importing the dataset
lines =  open('omim.txt').read().split('\n')
# There are a lot of whitespace lines which need to be removed.
lines = [line.strip() for line in lines if line.strip() != '']


#Cleaning the texts

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text) 
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"it's","it is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)#Punctuations probably don't need to be removed if we 
    #are going to consider context.
    return text


#test_line = lines[4]
# Since it is a test , will only try on 100,000 entries.
#This is a very slow process..
#TODO , could be parallelized   
corpus = []
for i in range(0,5000):
    line = re.sub('[^a-zA-Z]',' ',lines[i])
    line = line.lower()
    line = line.split()
    #Removing non significant words that don't have a sentiment attached.
    #Stemming (grouping words like loved,love,loving etc) only consider root of word.
    #We don't want a too sparse matrix
    ps = PorterStemmer()
    line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
    #Joining words back
    line = ' '.join(line)
    corpus.append([clean_text(line)])

############### Part 2 - Creating a Word2Vec model ###############
#The corpus needs to be a list of lists
model = Word2Vec(corpus,workers=8)
words = list(model.wv.vocab)
print(words)
print(model['gene function'])
model.most_similar_cosmul(positive=['gene function'])