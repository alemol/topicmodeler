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

import gensim.models
from gensim import corpora

import configparser
import os
import sys
import getopt
from os.path import (isfile, isdir, join, exists)
import glob
import ast
from ast import literal_eval
from operator import itemgetter
from shutil import rmtree
from pprint import pprint
import logging
import simplejson as json

#logging.basicConfig(format='DEBUG: %(message)s', level=logging.DEBUG)
logging.basicConfig(format='INFO: %(message)s', level=logging.INFO)


def fit_lda_model(corpus, dictionary, cfg):
    #  See documentation for full options
    #  https://radimrehurek.com/gensim/models/ldamulticore.html
    model = gensim.models.ldamulticore.LdaMulticore(corpus, id2word=dictionary,**cfg)

    return model

def store_model(m, cfg, out_dir):
    models_dir = mkdir(out_dir, 'modelos')
    fname = join(models_dir, 'lda_model')
    try:
        m.save(fname)
    except Exception as e:
        raise e

def store_wordtopics(m, cfg, out_dir):
    topics_dir = mkdir(out_dir, 'topics')
    for k in range(cfg['num_topics']):
        with open(join(topics_dir, 'topic_'+str(k)+'.csv'), 'w') as csvfile:
            for (term, prob) in m.show_topic(k, topn=50):
                csvfile.write('"{}",{:0.7f}\n'.format(term, prob))


def store_doctopics(input_documents, m, corpus, titles, cfg, out_dir):
    docs_dir = mkdir(out_dir, 'clusters')

    num_topics = cfg['num_topics']
    top_dict = {t : (None, 0.0) for t in range(num_topics)}
    doc_counts = {t : [] for t in range(num_topics)}

    for (i, bow_doc) in enumerate(corpus):

        probs_topics = m.get_document_topics(bow_doc)
        # select max prob topic from lists like [(2, 0.86289364), (4, 0.13553241)]
        # WARNING: ignoring when bow_doc is [] or probs_topics []
        # this causes less documents un table
        if not probs_topics:
            continue

        topic, prob = max(probs_topics, key=itemgetter(1))

        # update hash tables
        doc_counts[topic].append((titles[i], prob))
        if top_dict[topic][1] < prob:
            top_dict[topic] = (titles[i], prob)

        dtext = get_content(input_documents, titles[i])

    with open(join(docs_dir, 'list_clusters.csv'), 'w') as file:
        file.write('topic,ndocs,maxprob,text\n')
        for t, l in doc_counts.items():
            if len(l) > 0:
                sorted_l = sorted(l, key = lambda x : x[1], reverse=True)
                with open(join(docs_dir, 'cluster_{}.csv'.format(t)), 'w') as k:
                    for title, prob in sorted_l:
                        text = get_content(input_documents, title)
                        k.write('{},{},{:0.6f},{}\n'.format(t,title,prob,text))

                text = get_content(input_documents, top_dict[t][0])
                file.write('{},{},{},{}\n'.format(
                    t,
                    len(l),
                    sorted_l,
                    text))


def store_cluster(input_documents, m, corpus, vocab, titles, cfg, out_dir):
    # store each topic in a separate .csv file
    store_wordtopics(m, cfg, out_dir)
    store_doctopics(input_documents, m, corpus, titles, cfg, out_dir)
    #store_clustopics(m, corpus, vocab, titles, cfg, out_dir)
    return

def coherence(m, corpus):
    coherence_list = [coh_score 
        for (_, coh_score) in m.top_topics(corpus)]
    logging.info('coherence_ {}'.format(coherence_list[:5]))

def load_corpus(dir_inputvecspace):
    # warning: could match many files but keeping only the first
    path_corpus = glob.glob(join(dir_inputvecspace,'*corpus*mm')).pop()
    #print('path_corpus', path_corpus)
    path_dictionary = glob.glob(join(dir_inputvecspace,'*diccionario*dict')).pop()
    #print('>', dir_inputvecspace)
    #print('>',glob.glob(join(dir_inputvecspace,'*tulos*txt')))
    path_titles = glob.glob(join(dir_inputvecspace,'*tulos*txt')).pop()
    corpus = corpora.MmCorpus(path_corpus)
    vocab = corpora.Dictionary.load(path_dictionary)

    with open(path_titles) as f:
        titles = ast.literal_eval(f.read())

    return (corpus, vocab, titles)


# UTILS
def get_content(docs_path, basename):
    #docs_path = '../anaestvspace/strain/'
    file_path = join(docs_path, basename)
    if file_path.endswith('json'):
        try:
            with open(file_path, 'r') as f_read:
                j = json.loads(f_read.read())
        except:
            raise

        header_str = '{} - {}...'.format(
            # j['medium'],
            # j['date'],
            j['header'],
            j['transcripcion'][:50]
            )

        return ' '.join([part for part in header_str.split(' ') if part]).replace('\n', '')
    else:
        with open(file_path, 'r') as f_read:
            content = f_read.read()
        return (content.replace('\n', ' '))[:50]

def readconfig(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    cfg = dict()
    for k, v in config._sections['LDA'].items():
        try:
            value = literal_eval(v)
        except ValueError:
            value = v
        cfg[k]=value

    logging.info(cfg)
    return cfg


def mkdir(out, name):
    create_dir = join(out, name)
    if os.path.exists(create_dir):
        rmtree(create_dir)
    os.makedirs(create_dir)
    return create_dir


if __name__ == '__main__':

    usage = 'Usage: ldaclusters.py -i <input_vecspace> -d <doc_dir> -c <config_file> -o <out_dir> -h <help>'
    if not sys.argv[1:]:
        print(usage)
        sys.exit(2)

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'i:d:c:o:h',
                                   ['input_vecspace',
                                    'doc_dir',
                                    'config_file',
                                    'out_dir',
                                    'help'])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif opt in ('-i', '--input_vecspace'):
            input_vecspace = arg
        elif opt in ('-d', '--doc_dir'):
            doc_dir = arg
        elif opt in ('-c', '--config_file'):
            config_file = arg
        elif opt in ('-o', '--out_dir'):
            out_dir = arg if arg else './'

    # read configuration
    cfg = readconfig(config_file)
    # load vector space of corpus 
    (corpus, vocab, titles) = load_corpus(input_vecspace)
    logging.info('Titles list {} . . .'.format(titles[:5]))

    # produce the latent dirichlet model
    m = fit_lda_model(corpus, vocab, cfg)

    # get coherence values
    coherence_list = coherence(m, corpus)

    # store models: lda, topics, doc, assignments
    store_model(m, cfg, out_dir)
    store_cluster(doc_dir, m, corpus, vocab, titles, cfg, out_dir)
