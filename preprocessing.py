# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sklearn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import RAKE
class preprocessing:
	def __init__(self):
		self.stop_word = set(stopwords.words("english"))
    def wordnet(self,words):
	    words =  WordNetLemmatizer().lemmatize(word)
	    return None if sentece in self.stop_word else sentence

    def tokenize(self,sentence):
	    sentence = re.sub('[^a-zA-Z1-9]',' ',sentence)
	    sentence = re.lower()
	    # sentence split into word
	    sentence = word_tokenize(sentence)
	    sentence = map(self.wordnet,sentence)
	    sentence = ' '.join([word for word in sentence])
        return sentence

    def classification(self,label):
	    label = map(word_tokenize,sentence)
	    label = label[,0:1]
	    label = list(map(' '.join,label))
	    return label

    def group_apply(self,label,text):
	    df = pd.DataFrame(label=label,text=text)
	    g = df.groupby("label")
	    keywords_list = g.apply(rake)
	    return keywords_list.flatten()

# raker for extracting keywords
    def raker(self,text):
	    rake = RAKE.Rake(RAKE.SmartStopList())
	    keywords =  rake.run(text,
		    minCharacters = 1, 
		    maxWords = 3, 
		    minFrequency = 1)
	    maxlength = max(len(text)/200,len(keywords))
	    vocabulary = np.array(keywords)[0:(maxlength+1),0]
	    return self.tf_idf(text,vocabulary) 


    def tf_idf(self,text,vocabulary):
	    max_features = len(text)/200
        vectorizer = TfidfVectorizer(analyzer = 'word',
    	    max_features = max_features, 
    	    min_df = 10, 
    	    vocabulary = vocabulary)
        vectorizer.fit_transform(text)
        ind = np.argsort(vectorizer.idf_)[0:len(vocabulary)//3]
        keywords = vectorizer.get_feature_names()[ind]
        return keywords
