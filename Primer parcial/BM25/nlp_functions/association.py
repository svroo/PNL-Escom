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

# Modificada
def get_contexts_sents(vocabulary, sents, window = 8):
    """
        Here you can get the context of certain word of your setences.
        vocabulary: unique words.
        sents: corpus by sentences.
        windows: size of the context for each word.
    """
    contexts = dict()
    for w in vocabulary:
        context = list()
        for sent in sents:
            for i in range(len(sent)):
                if sent[i] == w:
                    for j in range(i - int(window / 2), i):
                        if j >= 0:
                            context.append(sent[j])
                    try:
                        for j in range(i + 1, i + (int(window / 2) + 1)):
                            context.append(sent[j])
                    except IndexError:
                        pass
        contexts[w] = context
    return contexts
# Nueva
def bm25(vector, avdl, k = 0.25, b = 0.25):
    new_vector = np.divide((k+1) * vector, vector + k * (1 - b + (b * np.sum(vector) / avdl)))
    return new_vector
    
    
# Nueva
def get_idf(vectors):
    num_context = len(vectors)
    total_aparitions = [0 for i in range(num_context)]
    for v in vectors.values():
        i = 0
        for element in v:
            if element != 0:
                total_aparitions[i] = total_aparitions[i] + 1
            i = i + 1
    idf = list()
    for element in total_aparitions:
        if element != 0:
            idf.append(np.log((num_context + 1) / element))
        else:
            idf.append(element)
    return idf



# Modificada
def get_vectors(vocabulary, contexts):
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
        vectors[v] = vector
    dls = list()
    for v in vectors.values():
        dls.append(np.sum(v))
    avdl = np.sum(dls) / len(dls)
    for k, v in vectors.items():
        new_vector = bm25(v, avdl)
        s = np.sum(new_vector)
        if s != 0:
            new_vector = new_vector / s
        vectors[k] = new_vector
    return vectors


# Nueva
def s_bm25(word, idf, vectors, aux_path = ''):
    """
        Here you can get the similaritie of each word.
        word: the word you want to get similaritie.
        vectors: the vector space of your vocabulary.
        aux_path: the name you want to add to the similarities.
    """
    similarities = dict()
    v = np.multiply(idf, vectors[word])
    for w in vectors.keys():
        v2 = np.multiply(idf, vectors[w])
        similarities[w] = np.dot(v, v2)
    similarities = (sorted(similarities.items(), key = lambda item: item[1], reverse = True))
    with open('./salida/bm25/similar_to_' + word[0] + aux_path + '.txt', 'w', encoding = 'utf-8') as f:
        for item in similarities:
            f.write(str(item) + '\n')


# Usada
def s_dot_product(word, idf, vectors, aux_path = ''):
    """
        Here you can get the similaritie of each word.
        word: the word you want to get similaritie.
        vectors: the vector space of your vocabulary.
        aux_path: the name you want to add to the similarities.
    """
    similarities = dict()
    v = np.multiply(vectors[word], idf)
    for w in vectors.keys():
        v2 = np.multiply(vectors[w], idf)
        similarities[w] = np.dot(v2, v)
    similarities = (sorted(similarities.items(), key = lambda item: item[1], reverse = True))
    with open('./dot_product/similar_to_' + word[0] + aux_path + '.txt', 'w', encoding = 'utf-8') as f:
        for item in similarities:
            f.write(str(item) + '\n')

# Usada
def s_cosine(word, idf, vectors, aux_path = ''):
    """
        Here you can get the similaritie of each word.
        word: the word you want to get similaritie.
        vectors: the vector space of your vocabulary.
        aux_path: the name you want to add to the similarities.
    """
    similarities = dict()
    v = np.multiply(vectors[word], idf)
    for w in vectors.keys():
        v2 = np.multiply(vectors[w], idf)
        if np.linalg.norm(v) == 0 or np.linalg.norm(v2) == 0:
            similarities[w] = 0
        else:
            similarities[w] = np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2))
    similarities = (sorted(similarities.items(), key = lambda item: item[1], reverse = True))
    with open('./cosine/similar_to_' + word[0] + aux_path + '.txt', 'w', encoding = 'utf-8') as f:
        for item in similarities:
            f.write(str(item) + '\n')

# Modificada
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
    tag = word[1]
    new_vectors = dict()
    for k, v in vectors.items():
        if k[1] == tag:
            new_vectors[k] = v
    if dot_product:
        s_dot_product(word, new_vectors, aux_path)
                
    if cosine:
        s_cosine(word, new_vectors, aux_path)
        
# Modificada
def similar_words_better(word, vectors, idf, vocabulary, aux_path = '', tf_idf = False, dot_product = False, cosine = False):   
    """
        The Pipline to get similarities.
        word: the word you want to get similaritie.
        vectors: the vector space of your vocabulary.
        aux_path: the name you want to add to the similarities.
        dot_product: boolean, if you want to get similarities by this method.
        cosines: boolean, if you want to get similarities by this method.
        tf_idf: boolean, if you want to get similarities by this method.
    """   
    idf = get_idf(vectors)
    words = list()
    for v in vocabulary:
        if v[0] == word:
            words.append(v)
    for word in words:
        tag = word[1]
        new_vectors = dict()
        for k, v in vectors.items():
            if k[1] == tag:
                new_vectors[k] = v
        if dot_product:
            s_dot_product(word, idf, new_vectors, aux_path)

        if cosine:
            s_cosine(word, idf, new_vectors, aux_path)
            
        if tf_idf:
            s_bm25(word, idf, new_vectors, aux_path)