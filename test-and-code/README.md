Application of LDA on Brown Corpus
========
In recent years, Data Mining is a very active research area, not just in the domain of Computer Science. In social science, researchers are trying taking a data-driven approach on traditional problems. (i.e. topic model for data discovery and empirical validation).

[Latent Dirichlet Allocation (LDA)](http://machinelearning.wustl.edu/mlpapers/paper_files/BleiNJ03.pdf) is an unsupervised algorithm which can allocate a topic distribution on documents in a fixed collection. Some researchers adapts this topic model to measure an external variable of interest and gains worth-noting achievements. [a survey](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf) However, this method must be carefully validated.

Here, we apply LDA using Gibbs sampling on Brown Corpus, which has been categorized, and compare the labelled categories and generated topics.


Dependence
------------

```bash
$ pip install numpy
$ pip install lda
$ pip install nltk
$ pip install sklearn

$ python setup.py install
```

Usage
-----

```python
from lda_topic.topic import *
lm = LDA_model(brown, True) #True: use Porter Stemmer (default)
lm.train(num_topic=15, num_iter=1000) #Brown corpus has 15 categories
lm.topic_kw() #show top-8 words of all topics
teston(lm.location['government'][0], lm.location['government'][1], 'government')  #show results for 'government' passages					
```

Result (stemmed)
---

```python
Topic 0: thi hi ha onli life man time ani
Topic 1: use thi water time cut make food inch
Topic 2: wa hi said like look thi man did
Topic 3: af wa thi use measur surfac form number
Topic 4: wa men road feet land street gun river
Topic 5: thi develop school need problem use work area
Topic 6: american nation ha war presid new world state
Topic 7: mrs year ha music play wa new hi
Topic 8: wa hi church said new state john law
Topic 9: state year thi cost 000 ani unit busi
```
15 Categories: adventure, belles_lettres, editorial, fiction, government, hobbies, humor, learned, lore, mystery, news, religion, reviews, romance, science_fiction

The algorithm discovers the information relating to government, hobbies (cook), and humor (musical). See Topic 6-9.

However, the algorithm is distracted by some verbs. The top words corresponding to some categories (fiction, lore, etc) are not very meaningful.


To-do
---
* __word removal__ - In the test, we use Porter Stemmer which results in a 28085-size vocabulary (raw text has a size of 42090). However, further feature selection is in need, i.e. removal of some verbs
