from glob import glob
import re
import string
import funcy as fp
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
import nltk
import pandas as pd
import pyLDAvis.gensim as gensimvis
import pyLDAvis

FILTER_REGEX = re.compile(r"[^a-z '#]")
TOKEN_MAPPINGS = [(FILTER_REGEX, ' ')]


def tokenize_line(line):
    res = line.lower()
    for regexp, replacement in TOKEN_MAPPINGS:
        res = regexp.sub(replacement, res)
    return res.split()


def tokenize(lines, token_size_filter=2):
    tokens = fp.mapcat(tokenize_line, lines)
    return [t for t in tokens if len(t) > token_size_filter]


def load_doc(filename):
    group, doc_id = filename.split('/')[-2:]
    with open(filename, errors='ignore') as f:
        doc = f.readlines()
    return {'group': group,
            'doc': doc,
            'tokens': tokenize(doc),
            'id': doc_id}


docs = pd.DataFrame(list(map(load_doc, glob('data/hackathon/*/*')))).set_index(['group', 'id'])
docs.head()

def nltk_stopwords():
    return set(nltk.corpus.stopwords.words('english'))

def prep_corpus(docs, additional_stopwords=set(), no_below=5, no_above=0.5):
  print('Building dictionary...')
  dictionary = Dictionary(docs)
  stopwords = nltk_stopwords().union(additional_stopwords)
  stopword_ids = map(dictionary.token2id.get, stopwords)
  dictionary.filter_tokens(stopword_ids)
  dictionary.compactify()
  dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
  dictionary.compactify()

  print('Building corpus...')
  corpus = [dictionary.doc2bow(doc) for doc in docs]

  return dictionary, corpus

dictionary, corpus = prep_corpus(docs['tokens'])
MmCorpus.serialize('newsgroups.mm', corpus)
dictionary.save('newsgroups.dict')

lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, passes=10)

lda.save('newsgroups_50_lda.model')

# The optional parameter T here indicates that HDP should find no more than 50 topics
# if there exists any.
hdp = models.hdpmodel.HdpModel(corpus, dictionary, T=50)

hdp.save('newsgroups_hdp.model')

vis_data = gensimvis.prepare(hdp, corpus, dictionary)
pyLDAvis.display(vis_data)