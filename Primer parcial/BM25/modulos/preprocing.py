import re
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import PlaintextCorpusReader
from bs4 import BeautifulSoup
import re
from pickle import load
import numpy as np

stop_word_path = r"C:\Users\hp\Documents\VSCode\PNL-Escom\Primer parcial\Ejercicios en clase\lemmas and others\stopwords_es.txt"
corpus_path = "C:\\Users\\hp\\Documents\\VSCode\\PNL-Escom\\Primer parcial\\EXCELSIOR_100_files\\"

def get_text(path = corpus_path):
    '''
    Here you can get the all text from paht: EXCELSIOR_100_files, and you have the text without html target
    path = with defautl value
    '''
    
    # Obtenemos el corpus del directorio
    corpus = PlaintextCorpusReader(path, '.*')
    file_list = corpus.fileids()

    # Juntamos todo el corpus del directorio
    all_text = ''
    for file in file_list:
        with open (path + file, encoding = 'utf-8') as rfile:
            text = rfile.read()
            all_text += text
    
    # Quitamos etiquetas html
    soup = BeautifulSoup(all_text, 'lxml')
    clean_text = soup.get_text().lower()

    return clean_text

def tokenize_text(text):
    words = text.split()
    alphabetic_words = []
    for word in words:
        token = []
        for character in word:
            if re.match(r'^[a-záéíóúñü+$]', character):
                token.append(character)
        token = ''.join(token)
        if token != '':
            alphabetic_words.append(token)

    # Quitamos las Stop words...
    with open(r'C:\Users\hp\Documents\VSCode\PNL-Escom\Primer parcial\Ejercicios en clase\lemmas and others\stopwords_es.txt', encoding = 'utf-8') as f:
        stop_words = f.readlines()
        stop_words = [w.strip() for w in stop_words]

    final_words = [word for word in alphabetic_words if word not in stop_words]

    return final_words

def get_context(text, word, window = 8):
    '''
    Here you can get context of word
    text: list of words
    word: a word to wich similarity is measure of all other words in text
    '''

    vocabulary = sorted(list(set(text)))
    contexts={}
    for w in vocabulary:
        context = []
        for i in range(len(text)):
            if text[i] == w:
                for j in range(i -int(window/2), i):
                    if j >= 0:
                        context.append(text[j])
                try:
                    for j in range(i +1, i+(int(window/2))+1):
                        context.append(text[j])
                except IndexError:
                    pass
        contexts[w] = context

    return contexts


def create_frecuancy_vector(words, word):
    '''
    Here you can create frecuancy vector of yours words
    words: all text
    word: a word to wich similarity is measure of all other words in text
    '''
    # Obtenemos el vocabulario
    vocabulary = sorted(list(set(words)))

    # Creamos el diccionario
    vectors = {}
    for v in vocabulary:
        try:
            contexts = get_context(text= words, word= v)
            context = contexts[v]
        except KeyError:
            pass

        vector = []

        for voc in vocabulary:
            vector.append(context.count(voc))

        vector = np.array(vector)
        vectors[v] = vector

    return vectors

def BM25(corpus, word):

    k = 1.2
    b = 0.8
    contexto = create_frecuancy_vector(corpus, word)
    res = {}
    
    if word in contexto.keys():
        for w in contexto.keys():
            dl = len(contexto[w])
            suma = np.sum(len(contexto.values()))
            avdl = suma / len(contexto.keys())
            V = np.array(contexto[w])
            BM25_v = np.divide((k+1)*V, V+k*(1-b+(b*dl)/avdl))
            suma = np.sum(BM25_v)
            BM25_v = BM25_v / suma
            res[w] = np.array(BM25_v)
    else:
        print(f'La palabra "{word}", no se encuentra en el corpus')

    res = (sorted(res.items(), key=lambda item: item[1], reverse=True))
    with open(file= "C:\\Users\\hp\\Documents\\VSCode\\PNL-Escom\\Primer parcial\\BM25\\salida\\bm25\\" + "bm25_without_IDF_for_" + word + '.txt', mode = 'w', encoding='utf-8') as f:
        for item in res:
            f.write(str(item)+'\n')

def main():
    text = get_text()
    text = tokenize_text(text)

    word = 'empresa'
    BM25(text, word)

if __name__ == '__main__':
    main()

