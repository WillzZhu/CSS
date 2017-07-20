# -*- coding: utf-8 -*-

import numpy as np
import lda


'''
加载reuters的新闻，X.shape 输出维度(395, 4258)
'''
X = lda.datasets.load_reuters()
print(X.shape)
print(X[0][:100])           #第一篇新闻对应的前100个vocabulary单词出现频率



'''
vocabulary: 包含了"不同"的单词，总计4258个
因此，X中包含了395篇路透社新闻，以长度为4258的"词袋"来表示某篇文章的单词出现频率
'''
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()



"""
初始化LDA模型，指定我们要寻找的话题量是20，迭代1500次
"""
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)

"""
开始拟合
"""
model.fit(X)  # model.fit_transform(X) is also available



"""
将每篇文章最具代表性的前8个话题打印出来
"""
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

"""
输出：
Topic 0: british churchill sale million major letters west
Topic 1: church government political country state people party
Topic 2: elvis king fans presley life concert young
Topic 3: yeltsin russian russia president kremlin moscow michael
Topic 4: pope vatican paul john surgery hospital pontiff
Topic 5: family funeral police miami versace cunanan city
Topic 6: simpson former years court president wife south
Topic 7: order mother successor election nuns church nirmala
Topic 8: charles prince diana royal king queen parker
Topic 9: film french france against bardot paris poster
Topic 10: germany german war nazi letter christian book
Topic 11: east peace prize award timor quebec belo
Topic 12: n't life show told very love television
Topic 13: years year time last church world people
Topic 14: mother teresa heart calcutta charity nun hospital
Topic 15: city salonika capital buddhist cultural vietnam byzantine
Topic 16: music tour opera singer israel people film
Topic 17: church catholic bernardin cardinal bishop wright death
Topic 18: harriman clinton u.s ambassador paris president churchill
Topic 19: city museum art exhibition century million churches
"""
