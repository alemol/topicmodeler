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


from tmcore.vspace import (load_fromfiles, vectorize, vect2gensim, store_vspace)
from tmcore.ldaclusters import (load_corpus, fit_lda_model, coherence, store_model, store_cluster)
from tmcore.cloud import (cloudify)
from tmcore.topicassign import(get_analyzer, get_dictionary, eval_documents)
from tmcore.utils import (in_cache, update_cache, readconfig)
# modulos externos
import gensim.models


def vectorizer(input_docs, max_df, min_df, features, range_ngram,
               config_file='resources/config.cfg', out_dir='out/vspace'):
    """
    vectorizer

    función para generar un modelo a partir de la vectorización de input_docs.

    Args:

        input_docs: La ruta de un directorio o de un archivo con los textos.

        max_df: El valor umbral de frecuencia máxima de término a partir del cual
        se debe comenzar a excliur, por ejemplo 0.99 indica la exclusión de 
        ngramas que aparecen en el 99% de los textos.

        min_df: El valor umbral de frecuencia mínima de término a partir del cual
        se debe comenzar a excliur, por ejemplo 0.01 indica la exclusión de
        ngramas que aparecen en el 1% de los textos.

        features: El valor máximo de features en la vectorización, por ejemplo
        10000 indica que los vectores de las bolsas de palabras pueden contener
        como máximo 10000 valores.

        range_ngram: El número máximo de palabras en los n-gramas, por ejemplo 3
        indica que se deben generar n-gramas de tamaño 1,2 y 3.

        config_file: La ruta del archivo de parametrización.

        out_dir: La ruta del directorio donde se generarán los archivos de salida.

    Dentro del directorio especificado en el argumento out_dir de vectorizer()
    se generarán los siguientes archivos:

        corpus.mm: Un vaciado a archivo de la instancia del objeto MmCorpus al
        ser serializado.

        corpus.mm.index: Archivo auxiliar de corpus.mm para hacer más eficientes
        los algoritmos que usan el objeto MmCorpus (se genera automáticamente).

        diccionario.dict: Un vaciado a archivo de la instancia del objeto
        Dictionary de gensim, contiene un mapeo entre ngramas y el ids de los
        ngramas Ver documentación.

        matriz.pkl: Matriz documento-término generada por el vectorizador de
        frecuencia de término CountVectorizer Ver documentación.

        palabras_descartadas.txt: La lista exacta de n-gramas que fueron excluidos en la generación del modelo vectorial.

        tdiccionario.txt: Listado en modo texto del contenido del diccionario
        generado por la instancia del objeto Dictionary de gensim.

        títulos: Los títulos de los documentos procesados.

        vectorizadorTF.pkl: Un vaciado binario de la instancia del objeto
        CountVectorizer luego de haber procesado los documentos mediante el
        método fit_transform(documents) Ver documentación.

        vocabulario.pkl: Serializado de la lista de n-gramas incluidos en los
        vectores del espacio vectorial.

    """
    (docs, titles) = load_fromfiles(input_docs, out_dir)
    # carga config desde archivo y por parámetros
    cfg = readconfig(config_file, 'VSPACE')
    args = {'documents':docs,
            'max_df':max_df,
            'min_df':min_df,
            'n_features':features,
            'out_dir':out_dir,
            'ngram':range_ngram}
    cfg.update(args)

    # produce objetos de vectorizado y matriz términ-documento
    (tf_vectorizer, dtmatrix) = vectorize(cfg)
    # transforma sparse matrix en gensim corpus
    (gensim_corpus, gensim_dict) = vect2gensim(tf_vectorizer, dtmatrix)
    # vaciado de objectos en archivos
    store_vspace(tf_vectorizer, dtmatrix, gensim_corpus, gensim_dict, out_dir)


def topicsfinder(input_docs, num_topics, iterations, passes,
                 vecspace_dir='out/vspace', config_file='resources/config.cfg',
                 lda_dir='out/lda'):
    """
    topicsfinder

    función para generar un modelo probabilístico de temas en documentos
    basado en Latent Dirichlet Allocation.

    Args:

        input_docs: La ruta del directorio con los textos, se usa para obtener
        los nombres.

        num_topics: El número de temas al que se quiere ajustar el modelo.

        iterations: El número máximo de iteraciones en el corpus en la inferencia de la distribución de topics.

        passes: El número de pasadas a través del corpus durante el
        entrenamiento (puede alentar para números grandes de documentos.

        vecspace_dir: La ruta del directorio donde se generaron los archivos de
        salida generados por vectorizer() en la etapa anterior.

        config_file: La ruta del archivo de parametrización.

        lda_dir: La ruta del directorio donde se generarán los archivos de salida.

    Dentro del directorio especificado en el argumento lda_dir de topicsfinder()
    se generarán los siguientes directorios y archivos:

        clusters/cluster_i.csv: Contiene una tabla csv por cada cluster i
        en formato: clusterID,docID,probabilidad,textito

        clusters/list_clusters.csv: contiene una tabla con información de todos
        los clusters en formato: topic,ndocs,maxprob,text

        modelos/: Contiene todos los archivos generados al serializar una
        instancia gensim.models.ldamulticore.LdaMulticore mediante el método save.

    """
    # carga config desde archivo y por parámetros
    cfg = readconfig(config_file, 'LDA')
    args = {'num_topics':num_topics,
            'iterations':iterations,
            'passes':passes}
    cfg.update(args)

    # carga objetos del vectorizado
    (corpus, vocab, titles) = load_corpus(vecspace_dir)

    # ajuste por latent dirichlet allocation
    m = fit_lda_model(corpus, vocab, cfg)
    # get coherence values
    coherence_list = coherence(m, corpus)

    # vaciado de objectos en archivos lda, topics, doc, assignments
    store_model(m, cfg, lda_dir)
    store_cluster(input_docs, m, corpus, vocab, titles, cfg, lda_dir)


def cloudyfier(n_clouds, n_words,
               lda_file='out/lda/modelos/lda_model', cloud_dir='out/wclouds'):
    """
    cloudyfier

    función para generar nubes de palabras de temas basado en las probabilidades
    de Latent Dirichlet Allocation.

    Args:

        n_clouds: El número de nubes que se generarán en imagen, debe ser
        igual o mayor al número de temas pero se pueden generar menos nubes
        que el número de temas de LDA. El orden de generación es el orden de
        significancia de los topics.

        n_words: El número de palabras al que se quiere las nubes.

        lda_file: La ruta del archivo del modelo LDA.

        cloud_dir: La ruta del directorio donde se generarán los archivos de salida.

    Dentro del directorio especificado en el argumento cloud_dir de cloudyfier()
    se generarán los siguientes archivos:

        wclouds/wordcloud_i.png: Contiene un archivo png por cada distribución i
        de probabilidades de ngramas.

    """
    lda = gensim.models.ldamulticore.LdaMulticore.load(lda_file, mmap='r')
    list_of_topics = lda.show_topics(num_topics=n_clouds,
                                     num_words=n_words,
                                     log=False,
                                     formatted=False)
    cloudify(list_of_topics, cloud_dir)


def topics_assigner(input_docs,
                    path_vectorizer='out/vspace/vectorizadorTF.pkl',
                    path_dict='out/vspace/diccionario.dict',
                    path_model='out/lda/modelos/lda_model'):
    """
    topics_assigner

    función para asignar documentos a temas encontrados en un modelo LDA 
    entrenado.

    Args:

        input_docs: La ruta del directorio con los documentos a asignar.

        path_vectorizer: La ruta al archivo del objeto de vectorizador de sklearn.

        path_dict: La ruta al archivo del objeto de Dictionary de gensim

        path_model: La ruta al archivo del objeto de modelo LDA entrenado

    Return:

        str: una cadena en formato

        doc_id    [(topic_i, prob_i),...]

        tal que la lista es ordenada por relevancia y solamente muestra elementos
        con probabilidades por encima de un valor umbral. por ejemplo:

        la_jornada-80317421.json   [(10, 0.96167856)]
        la_jornada-80317608.json    [(7, 0.244412), (10, 0.29511043), (11, 0.3670631), (12, 0.060325574)]

    """
    # Load a pretrained vectorizer from disk in order to retrieve the same preprocesing callable.
    text_analyzer = get_analyzer(path_vectorizer)

    # Load a pretrained gensim Dictionary from disk.
    trained_dictionary = get_dictionary(path_dict)

    # Load a pretrained LDA model from disk.
    lda = gensim.models.ldamulticore.LdaMulticore.load(path_model, mmap='r')

    # Get topic probabilities of LDA trained model for every unseen (not used during train) text in documents_dir
    eval_documents(input_docs, text_analyzer, trained_dictionary, lda)


# Para probar la funcionalidad y mostrar todo el pipeline con datos de prueba
if __name__ == '__main__':

    #######  Generar Espacio Vectorial  #######

    if in_cache('vspace'):
        print('Espacio Vectorial en caché')
    else:
        vectorizer('data', 0.7, 0.3, 1000, 3)

    #######  Generar Modelo LDA  #######

    if in_cache('vspace') and in_cache('lda'):
        print('Modelo LDA en caché')
    else:
        #topicsfinder('data/Sinaloa', 3, 15000, 5)
        topicsfinder('data', 3, 5000, 5)

    # #######  Nubes de Palabras  #######

    if in_cache('vspace') and in_cache('lda') and in_cache('wclouds'):
        print('Nubes en caché')
    else:
        #cloudyfier(3, 2000)
        cloudyfier(3, 1600)

    #######  Asignación de temas a nuevas notas  #######
    #print('Asignación de temas')
    #topics_assigner('data/test')

