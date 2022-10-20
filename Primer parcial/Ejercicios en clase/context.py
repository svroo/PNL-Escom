
# Import the libraries needed for the module...
from ast import In
from os import execv
import numpy as np


def get_contexts(vocabulary, text, window = 8):
    """
        Here you can get the context of certain word of your corpus.
        vocabulary: unique words.
        text: corpus.
        windows: size of the context for each word.
    """
    contexts = dict()
    for w in vocabulary:
        context = list()
        for i in range(len(text)):
            if text[i] == w:
                for j in range(i - int(window / 2), i):
                    if j >= 0:
                        context.append(text[j])
                try: 
                    for j in range(i + 1, i + (int(window / 2) + 1)):
                        context.append(text[j])
                except IndexError:
                        pass
        contexts[w] = context
    return contexts


def get_contexts_sents(vocabulary, text, window = 8):
    """
        Here you can get the context of certain word of your setences.
        vocabulary: unique words.
        text: corpus by sentences.
        windows: size of the context for each word.
    """
    contexts = dict()
    for w in vocabulary:
        context = list()
        for sent in text:
            words = sent.split()
            for i in range(len(words)):
                if words[i] == w:
                    for j in range(i - int(window / 2), i):
                        if j >= 0:
                            context.append(words[j])
                    try:
                        for j in range(i + 1, i + (int(window / 2) + 1)):
                            context.append(words[j])
                    except IndexError:
                        pass
        contexts[w] = context
    return contexts


def get_vectors(vocabulary, contexts, prob = False):
    """
        Here you can create a vector space for the words of your vocabulary.
        vocabulary: unique words.
        contexts: all the contexts of each word.
    """
    vectors = dict()
    for v in vocabulary:
        context = contexts[v]
        vector = []
        for voc in vocabulary:
            vector.append(context.count(voc))
        vector = np.array(vector)
        if prob:
            s = np.sum(vector)
            if s != 0:
                vector = vector / s
        vectors[v] = vector
    return vectors


def s_dot_product(word, vectors, aux_path = ''):
    """
        Here you can get the similaritie of each word.
        word: the word you want to get similaritie.
        vectors: the vector space of your vocabulary.
        aux_path: the name you want to add to the similarities.
    """
    similarities = dict()
    v = vectors[word]
    for w in vectors.keys():
        similarities[w] = np.dot(vectors[w], v)
    similarities = (sorted(similarities.items(), key = lambda item: item[1], reverse = True))
    with open('./dot_product/similar_to_' + word + aux_path + '.txt', 'w', encoding = 'utf-8') as f:
        for item in similarities:
            f.write(str(item) + '\n')


def s_cosine(word, vectors, aux_path = ''):
    """
        Here you can get the similaritie of each word.
        word: the word you want to get similaritie.
        vectors: the vector space of your vocabulary.
        aux_path: the name you want to add to the similarities.
    """
    similarities = dict()
    v = vectors[word]
    for w in vectors.keys():
        v2 = vectors[w]
        if np.linalg.norm(v) == 0 or np.linalg.norm(v2) == 0:
            similarities[w] = 0
        else:
            similarities[w] = np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2))
    similarities = (sorted(similarities.items(), key = lambda item: item[1], reverse = True))
    with open('./cosine/similar_to_' + word + aux_path + '.txt', 'w', encoding = 'utf-8') as f:
        for item in similarities:
            f.write(str(item) + '\n')


def similar_words(word, vectors, aux_path = '', tf_idf = False, dot_product = False, cosine = False):   
    """
        The Pipline to get similarities.
        word: the word you want to get similaritie.
        vectors: the vector space of your vocabulary.
        aux_path: the name you want to add to the similarities.
        dot_product: boolean, if you want to get similarities by this method.
        cosines: boolean, if you want to get similarities by this method.
        tf_idf: boolean, if you want to get similarities by this method.
    """     
    if dot_product:
        s_dot_product(word, vectors, aux_path)
                
    if cosine:
        s_cosine(word, vectors, aux_path)

