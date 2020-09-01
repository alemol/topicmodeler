# -*- coding: utf-8 -*-
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
import sys
import getopt
import os
import re
import nltk
from os.path import isfile, isdir, join, exists
import dill as pickle
import simplejson as json
import gensim.models
from gensim.corpora.dictionary import Dictionary


def eval_documents(documents_dir, text_analyzer, trained_dictionary, lda):
    for root, directory, files in os.walk(documents_dir, topdown=True):
        for file in sorted(files, key=natural_keys):
            if file.endswith('txt') or file.endswith('json'):
                full_path = join(root, file)
                #print(full_path)
                text = load_txt(full_path) if file.endswith('txt') else load_json(full_path)
                text_as_list = text_analyzer(text)
                unseen_doc = trained_dictionary.doc2bow(text_as_list)

                # get topic probability distribution for unseen document
                vector = lda[unseen_doc]
                print('{}\t{}'.format(file, vector))

# getters  ----------------------------------

def get_analyzer(path_vectorizer):
    # Return a callable that handles preprocessing and tokenization.
    with open(path_vectorizer, 'rb') as fp:
        vectorizer = pickle.load(fp)

    return vectorizer.build_analyzer()

def get_dictionary(path_dict):
    # Load and return pretrained gensim Dictionary.
    with open(path_dict, 'rb') as fp:
        trained_dictionary = pickle.load(fp)

    return trained_dictionary

# UTILITIES ----------------------------------

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


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]



if __name__ == '__main__':

    usage = 'Usage: topic_assign.py -i <input_dir> -v <vectorizer_path> -d <dictionary_path> -l <ldamodel_path> -h <help>'
    if not sys.argv[1:]:
        print(usage)
        sys.exit(2)

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'i:v:d:l:h',
                                   ['input_dir',
                                    'vectorizer_path',
                                    'dictionary_path',
                                    'ldamodel_path',
                                    'help'])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif opt in ('-i', '--input_dir'):
            documents_dir = arg
        elif opt in ('-v', '--vectorizer_path'):
            path_vectorizer = arg
        elif opt in ('-d', '--dictionary_path'):
            path_dict = arg
        elif opt in ('-l', '--ldamodel_path'):
            path_model = arg

    # Load a pretrained vectorizer from disk in order to retrieve the same preprocesing callable.
    text_analyzer = get_analyzer(path_vectorizer)

    # Load a pretrained gensim Dictionary from disk.
    trained_dictionary = get_dictionary(path_dict)

    # Load a pretrained LDA model from disk.
    lda = gensim.models.ldamulticore.LdaMulticore.load(path_model, mmap='r')
    # TODO verify through assert isinstance(lda, gensim.models.ldamodel.LdaModel)

    # Get topic probabilities of LDA trained model for every unseen (not used during train) text in documents_dir
    eval_documents(documents_dir, text_analyzer, trained_dictionary, lda)

