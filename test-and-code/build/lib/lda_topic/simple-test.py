# -*- coding: utf-8 -*-

import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer



def test():
    count_vect = CountVectorizer(stop_words='english')
    X=count_vect.fit_transform(np.array(['I love swimming.',
                                        'Sun Yang is swimming this morning.',                                                #stemmer
                                        'He is a very good basketball player.',
                                        'In that basketball game, he took 34 scores.',
                                        'He started playing basketball since 9.',
                                        'Jogging is the beginning of my day.',
                                        'Many people went jogging on Monday.',
                                        'I go swimming today.'])).toarray()
    vocab = np.array(list(count_vect.vocabulary_.items()))
    model = lda.LDA(n_topics=3, n_iter=200, random_state=1)
    model.fit(X)
    topic_word = model.topic_word_
    n_top_words = 1
    for i, topic_dist in enumerate(topic_word):
        topic_words = []
        idx = np.argsort(topic_dist)[-n_top_words:-1]
        for j in idx:
            topic_words.append(vocab[np.where(vocab == str(j))[0][0]][0])
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    return topic_word, vocab
"""
Topic 0: basketball
Topic 1: swimming
Topic 2: jogging
"""
