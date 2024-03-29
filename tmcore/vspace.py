"""

El topic modeling core está concebido como un pipeline modular de tal manera que el primer input
del pipeline son las notas y el output depende de los módulos que se conecten,
o se desconecten, del pipeline. ANAEST permite reusar resultados parciales
porque los módulos se comunican a través de objetos en archivos.

Por ejemplo, un pipeline para vectorizar un conjunto de notas (X_Notas) sería:

X_Notas > Espacio Vectorial (X_Notas)

y un pipeline para generar clustering por tema sería:

X_Notas > Espacio Vectorial (X_Notas) > LDA (X_Notas)

Como los módulos se comunican a través de objetos en archivos, es posible usar los resultados parciales de un módulo y mandarlo a la entrada de otro módulo.
Por ejemplo, un pipeline para generar las nubes de palabras de un conjunto del cual ya tenemos el model LDA en archivos sería:

LDA (X_Notas) > WClouds (X_Notas)

Contar con los objetos de vectorización y LDA permite además encontrar notas de un conjunto de prueba (Y_Notas) qué coinciden con los temas producidos por otro conjunto de entrenamiento (X_Notas) con el cual se generó el modelo LDA:

Y_Notas > LDA (X_Notas) >  Asignación de probabilidades



Created by Alex Molina
Noviembre 2019

"""

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
    doc_norep = re.sub(r'\b(\w+)( \1\b)+', r'\1', doc_l, flags=re.U)
    doc_nonum = re.sub(r'\b(\d+)\b', '', doc_norep, flags=re.U)
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
        if input_documents.endswith('txt'):
            docs = [l.strip() for l in open(input_documents).readlines()]
        elif input_documents.endswith('json'):
            with open(input_documents, 'r') as j_read:
                j = json.loads(j_read.read())
            # TODO GENERIC 
            records = j['SOLICITUDES']['SOLICITUD']
            docs = []
            titles = []
            for i, r in enumerate(records):
                text = r['DescripcionSolicitud']
                folio = r['Folio']
                print(i,'\t',folio,'\t',r['FechaSolicitud'],text)
                docs.append(text)
                titles.append(folio)
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
        toks = tf_vectorizer.get_feature_names()
        freqs = dtmatrix.toarray().sum(axis=0)
        counts={t: freqs[i] for i, t in enumerate(toks)}
        for k,v in sorted(counts.items(), key=lambda x:x[1], reverse=True):
            out.write('{}\t{}\n'.format(v, k))

    # store the gensim dictionary to file
    pname = join(out_dir, 'diccionario.dict')
    gensim_dict.save(pname)
    pname = join(out_dir, 'tdiccionario.txt')
    gensim_dict.save_as_text(pname)

    # store gensim corpus to disk, for later use
    pname = join(out_dir, 'corpus.mm')
    corpora.MmCorpus.serialize(pname, gensim_corpus)

    #pname = join(out_dir, 'frecuencias.txt')
    #with open(pname, 'w') as out:
        #idtok_dict = gensim_dict.id2token
        #print('idtok_dict', type(idtok_dict), len(idtok_dict))
        #print('gensim_dict', type(gensim_dict), len(gensim_dict.cfs))
        #for tokid, freq in gensim_dict.cfs.items():
            #print(tokid, freq)
            #out.write('{}\t{}'.format(idtok_dict[tokid],freq))

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
