# -*- coding: utf-8 -*-

import sys
from os import (makedirs)
from os.path import (join, exists)
import getopt
import gensim.models
from wordcloud import WordCloud
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def topic2cloud(i, dict_topic, cloud_dir):

    if not exists(cloud_dir):
        makedirs(cloud_dir)
    # load logo
    #logo_mask = np.array(Image.open("./img/lupa.png"))
    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white",
                          width=1920,
                          height=1080,
                          #mask=logo_mask,
                          ).generate_from_frequencies(dict_topic)

    plt.imshow(wordcloud, interpolation='bilinear', aspect='auto')
    plt.axis("off")
    cdir = join(cloud_dir, 'wordcloud_{}.png'.format(i))
    plt.savefig(cdir)
    plt.close()


def plot_Nclouds(n_clouds, n_words, lda):
    # see https://radimrehurek.com/gensim/models/ldamulticore.html
    # print_topics()
    # show_topics
    list_of_topics = lda.show_topics(num_topics=n_clouds,
        num_words=n_words,
        log=False,
        formatted=False)

    for (t_id, wprobs) in list_of_topics:
        print(t_id)
        topic_dist = {w:p for (w, p) in wprobs}
        #print(topic_dist)
        topic2cloud(t_id, topic_dist, out_dir)


def cloudify(list_of_topics, cloud_dir):
    for (t_id, wprobs) in list_of_topics:
        print('{}, [{},{},{}...]'.format(t_id, wprobs[0],wprobs[1],wprobs[2]))
        topic_dist = {w:p for (w, p) in wprobs}
        topic2cloud(t_id, topic_dist, cloud_dir)


if __name__ == '__main__':

    usage = 'Usage: cloud.py -i <input_model> -n <num_topics> -o <out_dir> -h <help>'
    if not sys.argv[1:]:
        print(usage)
        sys.exit(2)

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'i:n:o:h',
                                   ['input_model',
                                    'num_topics',
                                    'out_dir',
                                    'help'])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage)
            sys.exit()
        elif opt in ('-i', '--input_model'):
            model_file = arg
        elif opt in ('-n', '--num_topics'):
            ntopics = int(arg)
        elif opt in ('-o', '--out_dir'):
            out_dir = arg if arg else './'

    # model_file = '../repo/anaestclusters/jclusters/modelos/lda200top_30000it_asymmetrical_autoet'
    lda = gensim.models.ldamulticore.LdaMulticore.load(model_file, mmap='r')
    list_of_topics = lda.show_topics(num_topics=ntopics,
                                     num_words=250,
                                     log=False,
                                     formatted=False)
    cloudify(list_of_topics, out_dir)

