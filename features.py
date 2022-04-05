from collections import OrderedDict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
import collections


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        freq = dict()
        for text in X:
         
         for tok in text.split():
           
           if tok in freq:
             freq[tok] +=1
           else: freq[tok] =1

        freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        self.bow = [w[0] for w in freq_sorted][:self.k]
        
        # # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
            #<YOUR CODE>

        feat = {k:0 for k in self.bow}
        for token in text.split():
          if token in feat:
            feat[token] += 1
        result = list(feat.values())
        return np.array(result, "float32")


    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()
        self.tokens = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        freq = dict()
        for text in X:
         
         for tok in text.split():
           
           if tok in freq:
             freq[tok] +=1
           else: freq[tok] =1

        freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        token_list = [w[0] for w in freq_sorted][:self.k]
        
        N = len(X)
       
        for text in X:
          text_split = text.split()
          for tok in text_split:
            if tok not in self.tokens:
              self.tokens[tok] = sum([1.0 for i in X if tok in i.split()])     
        
        for token in self.tokens:
          self.idf[token] = np.log(N/self.tokens[token])
        # fit method must always return self

        
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        text_split = text.split()
        l = len(text_split)
        
        tf_idf_dictionary = {k:0 for k in self.idf}
        for word in text_split:
          if word in self.idf:
            tf_idf_dictionary[word] = (self.tokens[word]/l)*self.idf[word]
        
        result = list(tf_idf_dictionary.values())
        if self.normalize == True:
          norm = np.linalg.norm(result)
          result = result/(norm+0.000000001)
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
