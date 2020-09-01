import os
import sys
import getopt
import re
import ast
import dill as pickle
from time import time
from os.path import isfile, isdir, join, exists

from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import nltk
import gensim
from gensim import corpora
from gensim.corpora.dictionary import Dictionary

import simplejson as json


# VECTORIZATION ----------------------------------

def vectorize(cfg):

    # stop words definition
    stopwords_file = cfg['exclude_words']
    stop_list = build_stoplist(stopwords_file, cfg['out_dir'])

    # Use tf (raw term count) features for LDA.
    tf_vectorizer = CountVectorizer(max_df=cfg['max_df'],
                                    min_df=cfg['min_df'],
                                    max_features=cfg['n_features'],
                                    ngram_range=(1, cfg['ngram']),
                                    stop_words=stop_list,
                                    preprocessor=my_preprocessor,
                                    )

    start = time()
    dtmatrix = tf_vectorizer.fit_transform(cfg['documents'])
    end = time()
    print("fit_transform vector model ... done in {0:0.3f} miliseconds".format((end - start) * 1000))
    return (tf_vectorizer, dtmatrix)


def vect2gensim(vectorizer, dtmatrix):
     # transform sparse matrix into gensim corpus and dictionary
    start = time()
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(dtmatrix, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
    end = time()
    print("Transform vector model to gensim format ... done in {0:0.3f} miliseconds".format((end - start) * 1000))

    return (corpus_vect_gensim, dictionary)

# PREPROCESSING ----------------------------------

def my_preprocessor(text):
    doc_l = text.lower()
    doc_norep = re.sub(r'\b(\w+)( \1\b)+', r'\1', doc_l)
    doc_nonum = re.sub(r'\b(\d+)\b', '', doc_norep)
    preprocessed_doc = ' '.join(nltk.word_tokenize(doc_nonum))

    return preprocessed_doc

def build_stoplist(stopwords_file, out_dir, lang='spanish', store=True):

    with open(stopwords_file) as f:
        stop_list = ast.literal_eval(f.read())

    stop_list.extend(get_stop_words(lang))
    stop_list = list(set(stop_list))
    stop_list.extend([w.title() for w in stop_list])
    stop_list.extend([w.upper() for w in stop_list])

    if store:
        sname = join(out_dir, "palabras_descartadas.txt")
        with open(sname, 'w') as out:
             out.write('{}'.format(stop_list))

    return stop_list

# LOAD DATA ----------------------------------

def load_fromfiles(input_documents, out_dir):

    if isfile(input_documents):
        docs = [l.strip() for l in open(input_documents).readlines()]
    elif isdir(input_documents):
        print('Exploring dir', input_documents)
        (docs, titles) = docs_as_list(input_documents)

        # creates
        if not exists(out_dir):
            os.makedirs(out_dir)

        pname = join(out_dir, "títulos.txt")
        with open(pname, 'w') as out:
             out.write('{}'.format(titles))
    else:
        print('Invalid input')
        print(usage)
        sys.exit(2)

    return (docs, titles)

def docs_as_list(documents_dir):
    docs = []
    titles = []
    for root, directory, files in os.walk(documents_dir, topdown=True):
        for file in sorted(files, key=natural_keys):
            if file.endswith('txt') or file.endswith('json') or file.endswith('csv'):
                full_path = join(root, file)
                print(full_path)
                d = load_json(full_path) if file.endswith('json') else load_txt(full_path)
                docs.append(d)
                titles.append(file)

    return (docs, titles)

def load_txt(txt_path):
    try:
        with open(txt_path) as f:
            contents = f.read()
        contents = contents.replace('\n', '')
    except:
        raise
    return contents

def load_json(json_path):
    try:
        with open(json_path, 'r') as j_read:
            j = json.loads(j_read.read())
    except:
        raise
    return j['transcripcion']

# STORE DATA ----------------------------------

def store_vspace(tf_vectorizer, dtmatrix, gensim_corpus, gensim_dict, out_dir):

    pname = join(out_dir, 'vectorizadorTF.pkl')
    with open(pname, 'wb') as out:
        pickle.dump(tf_vectorizer, out)

    pname = join(out_dir, 'matriz.pkl')
    with open(pname, 'wb') as out:
        pickle.dump(dtmatrix, out)

    pname = join(out_dir, 'vocabulario.txt')
    with open(pname, 'w') as out:
       out.write('{}'.format(tf_vectorizer.get_feature_names()))

    # store the gensim dictionary to file
    pname = join(out_dir, 'diccionario.dict')
    gensim_dict.save(pname)
    pname = join(out_dir, 'tdiccionario.txt')
    gensim_dict.save_as_text(pname)

    # store gensim corpus to disk, for later use
    pname = join(out_dir, 'corpus.mm')
    corpora.MmCorpus.serialize(pname, gensim_corpus)


# UTILITIES ----------------------------------

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]



if __name__ == '__main__':

    usage = 'Usage: vspace.py -i <input_docs> -x <max_df> -n <min_df> -f <features> -s <stop_words> -r <range_ngram> -o <out_dir> -h <help>'
    argv = sys.argv[1:]
    if not argv:
        print(usage)
        sys.exit(2)

    try:
        opts, args = getopt.getopt(argv,
                                   'i:x:n:f:s:r:o:h',
                                   ['input_docs',
                                    'max_df',
                                    'min_df',
                                    'features',
                                    'stop_words',
                                    'range_ngram',
                                    'out_dir',
                                    'help'])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif opt in ('-i', '--input_docs'):
            input_documents = arg
        elif opt in ('-x', '--max_df'):
            max_df = float(arg)
        elif opt in ('-n', '--min_df'):
            min_df = float(arg)
        elif opt in ('-f', '--features'):
            max_features = int(arg)
        elif opt in ('-s', '--stop_words'):
            stop_list = arg
        elif opt in ('-r', '--range_ngram'):
            ngram = int(arg)
        elif opt in ('-o', '--out_dir'):
            out_dir = arg if arg else './'

    (docs, titles) = load_fromfiles(input_documents, out_dir)

    # produce vectorial model and document term matrix
    (tf_vectorizer, dtmatrix) = vectorize(documents=docs, 
                                          max_df=max_df,
                                          min_df=min_df,
                                          n_features=max_features,
                                          out_dir=out_dir,
                                          ngram=ngram,
                                          stopwords_file=stop_list)

    # transform sparse matrix into gensim corpus
    (gensim_corpus, gensim_dict) = vect2gensim(tf_vectorizer, dtmatrix)

    # store produced objets into files under named files
    afix = '_{}docs_{}maxdf_{}mindf_{}ngrm'.format(len(docs),
                                                     max_df,
                                                     min_df,
                                                     ngram)

    store_vspace(tf_vectorizer, dtmatrix, gensim_corpus, gensim_dict, afix, out_dir)
