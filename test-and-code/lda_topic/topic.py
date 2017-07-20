# -*- coding: utf-8 -*-

import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import time

'''
{'news': [0, 44], 'reviews': [71, 88], 'mystery': [403, 427], 'lore': [141, 189], 'belles_lettres': [189, 264], 'romance': [462, 491], 'adventure': [433, 462], 'humor': [491, 500], 'religion': [88, 105], 'fiction': [374, 403], 'editorial': [44, 71], 'hobbies': [105, 141], 'learned': [294, 374], 'science_fiction': [427, 433], 'government': [264, 294]}
'''



class LDA_model(object):
    def __init__(self, corpus, stem):
        """
        - corpus: Brwon corpus; 15 categories: adventure, belles_lettres, editorial, fiction, government, hobbies, humor, learned, lore, mystery, news, religion, reviews, romance, science_fiction
        - document: An numpy array of shape (N, ) containing 500 documents in Brown corpus
        """
        self.document = np.array([''])
        self.corpus   = corpus
        self.stemmer  = PorterStemmer().stem
        self.location = {}
        cnt = 0
        cur = None
        for file_id in self.corpus.fileids():
            if cur == None:
                cur = corpus.categories(fileids=file_id)[0]
            elif cur != corpus.categories(fileids=file_id)[0]:
                self.location[cur].append(cnt)
            if self.location.get(corpus.categories(fileids=file_id)[0])==None:
                self.location[corpus.categories(fileids=file_id)[0]] = []
                self.location[corpus.categories(fileids=file_id)[0]].append(cnt)
            cur = corpus.categories(fileids=file_id)[0]
            cnt += 1
            words = self.corpus.words(fileids=file_id)
            """
            stem the words
            """
            w = list(words)
            if stem:
                for idx in range(len(w)):
                    w[idx] = self.stemmer(w[idx])
            self.document = np.append(self.document, ' '.join(w))
        self.document = self.document[1:]
        self.location['humor'] = [491, 500]
        self.count_vect = CountVectorizer(stop_words='english')
        self.X_tr = self.count_vect.fit_transform(self.document)
        self.vocab = np.array(list(self.count_vect.vocabulary_.items()))

        self.model = None


    def feature_extractor(self, X):
        """
        - X: an numpy array of shape (n, ) containing n sentences
        """
        for sents in range(X.shape[0]):
            pass
            X[idx] = self.stemmer(X[idx])

    def train(self, num_topic=15, num_iter=1500):
        self.model = lda.LDA(n_topics=num_topic, n_iter=num_iter, random_state=1)
        self.model.fit(self.X_tr)

    def plot_log(self):
        plt.plot(self.model.loglikelihoods_[5:])
        plt.savefig('D:/coding/CSS/test-and-code/'+time.strftime("%m-%d-%H-%M",time.localtime()))


    def topic_kw(self):
        n_top_words = 8
        topic_word = self.model.topic_word_
        for i, topic_dist in enumerate(topic_word):
            top_words = []
            idx = np.argsort(topic_dist)[-n_top_words:]
            for j in idx:
                top_words.append(self.vocab[np.where(self.vocab == str(j))[0][0]][0])
            top_words.reverse()
            print('Topic {}: {}'.format(i, ' '.join(top_words)))


def teston(s, t, cate):
    print(cate)
    doc_topic_test = lm.model.transform(lm.X_tr[s:t])
    label = doc_topic_test.argmax(axis=1)
    return doc_topic_test, label


if __name__ == '__main__':
    lm = LDA_model(brown, True)
    for num_t in [10, 15, 20]:
        lm.train(num_topic=num_t, num_iter=500)
        lm.topic_kw()
        lm.plot_log()
        for kw, value in lm.location.items():
            logprob, label = teston(value[0], value[1], kw)
            print(label)
            print(np.bincount(label))
