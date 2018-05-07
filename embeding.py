# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import gensim
from gensim.models.word2vec import Word2Vec
import tensorflow as tf

class Embeding:
    def __init__(self,condition,ship,name,category,brand,text):
        self.condition = condition
        self.ship = ship
        self.name = name
        self.category = category
        self.brand = brand
        self.text = text

    def word2vec(self):
        w2v = Word2Vec(size = 200,min_count = 10)
        w2v.build_vocab(self.text)
        num = w2v.corpus_count
        epochs = w2v.iter
        w2v.train(self.text,total_examples = num,epochs = epochs)
        return w2v

    def BOW(self,vocabulary):
        vectorizer = CountVectorizer(analyzer = 'word', 
    	    min_df = 10,
    	    vocabulary = vocabulary)
        one_hot_encoder = vectorizer.fit_transform(self.text)
        one_hot_encoder = one_hot_encoder.toarray()
        return one_hot_encoder

    def initial(self,vocabulary):
        w2v = self.word2vec(self.text)
        window = w2v.corpus_count
        bow = self.BOW(self.text,vocabulary)
        max_length = max(np.sum(bow,axis=1))
        return w2v,window,bow,max_length

# w2v,window,bow,max_length = self.initial(self.text,vocabulary)

    def embeding(self,w2v,element):
	    vec = []
	    for i in range(len(element)):
		    vec.append(element[ind]*w2v[ind])
	    return np.array(vec)

    def sentence2vec(self,bow,w2v,data=self.text):
        vectorizer = TfidfVectorizer(analyzer = 'word',vocabulary = vocabulary)
        weight = vectorizer.fit_transform(data).toarray()
	    vec = []
	    for i in range(bow.shape[0]):
		    embedings = self.embeding(w2v,bow[ind])
	        vec.append(np.dot(weight[ind],embedings))
	    return np.array(vec)

    def combine(self,bow_text,bow_name,w2v):
        text = self.sentence(bow_text,w2v,self.text)
        text = self.sentence(bow_name,w2v,self.name)
        category = np.array([sum([w2v[i] for i in category[j]]) for j in category])
        condition = np.array(pd.get_dummies(pd.Series(self.condition)))
	    ship = self.ship
	    brand = [w2v[i] for i in brand]
        combined_data = {'name':name,'category':category,'brand':brand,'description':text,
            'condition':condition,'ship':ship}
	    return pd.DataFrame(combined_data)
