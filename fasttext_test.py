#Importing the libraries
import numpy as np
import re
import time
from gensim.models import FastText
import pandas as pd
from time import gmtime, strftime
today = strftime("%Y-%m-%d %H:%M:%S", gmtime())
############### Part 1 - Data Preprocessing ###############
# Importing the dataset
dataset = pd.read_table('genemap_copy.txt',delimiter='\t', lineterminator='\n').fillna("")
#Cleaning the texts
"""
On examining the dataset, we can see that columns Sort,Month,Day ,Year 
and Confidence (deprecated - see note) are most
likely not useful in the Word2Vec model. Hence they will be dropped.
Deciding to drop Gene Name , the chemical names are hard to deal with
preprocessing wise at the moment. We will revist.
"""
#TODO revisit Gene Name processing,Phenotype preprocessing.
dataset_drop_smdyc = dataset.iloc[:,1:].drop(['Month','Day','Year','Confidence (deprecated - see note)'],axis=1)
"""
Thoughts on preprocessing the remaining columns.
Cyto Location: periods and dashes seems important. Probably should not preprocess it.
Gene Symbols: Lowercase and Uppercase have meaning here. Probably should not process. Commas could be removed.#Safe to chab
It is safe to change Gene symbol case
Gene Name: If it is just a name, it can be lowercased and commas removed.
Anything seperated by a comma is a word in the column Gene Name.
MIM Number: No processing needed.
Mapping Method: No processing needed, uppercase/lowercase have meaning.
Comments: lowercased and punctuations can be removed.
Phenotypes: Can be probably be lowercased, remove punctuations and {} brackets. 
            That should be okay for Phenotypes for now. Will need to be revisited.
Phenotype just remove comma, and split on whitespace
Mouse Gene Symbol: Can be left as is. Lowercase/Uppercase seem to have meaning.
Can Make Mouse Gene lowercase
"""
#Cleaning the texts
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\"#/@;:<>{}+=~|.?,]","",text)#Punctuations probably don't need to be removed if we
    #are going to consider context.
    return text

def preprocess_df(dataframe):
    for index,row in dataframe.iterrows():
        #https://stackoverflow.com/questions/31569384/set-value-for-particular-cell-in-pandas-dataframe-with-iloc/31569794
        dataframe.iloc[index, dataframe.columns.get_loc('Comments')] = clean_text(row['Comments'])
       #df_copy.iloc[index, df_copy.columns.get_loc('Phenotypes')] = clean_text(row['Phenotypes'])
        #df_copy.iloc[index, df_copy.columns.get_loc('Gene Name')] = row['Gene Name'].lower()#clean_text(row['Gene Name'])
        dataframe.iloc[index, dataframe.columns.get_loc('Gene Symbols')] = clean_text(row['Gene Symbols'])
        dataframe.iloc[index, dataframe.columns.get_loc('Mouse Gene Symbol')] = clean_text(row['Mouse Gene Symbol'])

    return dataframe

def clean_pheno_and_gene_name(text):
    text = text.lower()
    text = re.sub(r"[\"#/@;:<>{}+=~|.?]","",text)#Punctuations probably don't need to be removed if we
    #are going to consider context.
    return text

def preprocess_pheno_and_gene_name(dataframe):
    index_to_pheno = {}
    index_to_gene_name = {}
    for index,row in dataframe.iterrows():
        index_to_pheno[index] = clean_pheno_and_gene_name(row['Phenotypes']).split(',')
        index_to_gene_name[index] = clean_pheno_and_gene_name(row['Gene Name']).split(',')
    return index_to_pheno,index_to_gene_name

df_copy = dataset_drop_smdyc.copy()
index_to_pheno,index_to_gene_name = preprocess_pheno_and_gene_name(df_copy)
df_copy_wo_pheno_gn = df_copy.iloc[:,1:].drop(['Phenotypes','Gene Name'],axis=1)

gene_data_preprocessed = preprocess_df(df_copy_wo_pheno_gn)

############### Part 2 - Tokenizing the dataset ###############

def row_to_sentences(dataframe):
    columns = dataframe.columns.values
    corpus = []
    for index,row in dataframe.iterrows():
        sentence = ''
        for column in columns:
            sentence += ' '+str(row[column])
        corpus.append([sentence])
    return corpus
corpus = row_to_sentences(gene_data_preprocessed)

def add_gn_pt_to_corpus(corpus,index_to_pheno,index_to_gene_name):
    augmented_corpus = corpus[:]
    for i in range(len(corpus)):
        augmented_corpus[i]+= index_to_pheno[i]
        augmented_corpus[i]+= index_to_gene_name[i]
    return augmented_corpus

augmented_corpus = add_gn_pt_to_corpus(corpus,index_to_pheno,index_to_gene_name)
tokenized_corpus = [sentence[0].split() for sentence in augmented_corpus]
#Stop words need to be removed
#TODO this needs to be finished
def remove_stop_words(word):
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    #Joining words back
    text = ' '.join(text)
    return text
############### Part 2 - Creating a FastText model ###############
#The corpus needs to be a list of lists
model = FastText(tokenized_corpus,min_count=1)
words = len(list(model.wv.vocab))
model.save('fasttext0.01'+today)#0.01_d_year

# Comparisions with ThinkData
terms = ['alk','angiogenesis',
         'invasion','signaling',
         'aneurysm','telomere',
         'lyme','brca',
         'alzheimer','gene']
dataframe_data = {}
for term in terms:
    if term in set(model.wv.vocab):
        dataframe_data[term] = []
        sim_terms = model.wv.most_similar(positive=[term])
        for sim_term,_ in sim_terms:
            dataframe_data[term].append(sim_term)
term_df = pd.DataFrame(dataframe_data)


