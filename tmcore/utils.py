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
from shutil import copyfile
import filecmp
import configparser
from ast import literal_eval
from os.path import (join, exists)

vspace_cache = [ ('out/vspace/corpus.mm', '.cache/last_corpus.mm'),
                ('out/vspace/diccionario.dict', '.cache/last_diccionario.dict')]

lda_cache = [('out/lda/modelos/lda_model', '.cache/last_lda_model')]


# Utilidades para manejo de cache
def in_cache(stage):
    # listas de archivos para manejo de caché
    if stage is 'vspace':
        return exists('out/vspace')
    elif stage is 'lda':
        return exists('out/lda')
    elif stage is 'wclouds':
        return exists('out/wclouds')

def filesin_cache(stage):
    # listas de archivos para manejo de caché
    if stage is 'vspace':
        comparisons_list = vspace_cache
    elif stage is 'lda':
        comparisons_list = lda_cache

    # True si todos los pares de archivos en la lista son iguales
    try:
        return all(filecmp.cmp(f1, f2) for (f1, f2) in comparisons_list)
    except  (OSError, IOError) as e:
        return False

def update_cache(stage):
    if stage is 'vspace':
        update_list = vspace_cache
    elif stage is 'lda':
        update_list = lda_cache
    for (src, dst) in update_list:
        copyfile(src, dst)

# utilidades para archivo de configuración
def readconfig(config_path, secc):
    config = configparser.ConfigParser()
    config.read(config_path)

    cfg = dict()
    for k, v in config._sections[secc].items():
        try:
            value = literal_eval(v)
        except ValueError:
            value = v
        cfg[k]=value

    return cfg
