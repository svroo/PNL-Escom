from nltk.corpus import PlaintextCorpusReader
from bs4 import BeautifulSoup
import re
import nltk
from nltk.stem import SnowballStemmer
from collections import Counter


def normalize(path = './../EXCELSIOR_100_files/'):

    corpus = PlaintextCorpusReader(path, '.*')
    file_list = corpus.fileids()
    

    all_text = ''
    for file in file_list:
        with open(path + file, encoding = 'utf-8') as rfile:
            text = rfile.read()
            all_text += text

    soup = BeautifulSoup(all_text, 'lxml')
    clean_text = soup.get_text()
    clean_text = clean_text.lower()

    words = clean_text.split()
    alphabetic_words = []
    for word in words:
        token = []
        for character in word:
            if re.match(r'^[a-záéíóúñü+$]', character):
                token.append(character)
        token = ''.join(token)
        if token != '':
            alphabetic_words.append(token)

    with open('./stopwords_es.txt', encoding = 'utf-8') as f:
        stop_words = f.readlines()
        stop_words = [w.strip() for w in stop_words]
    final_words = [word for word in alphabetic_words if word not in stop_words]
    print('Fin de la normalización...')
    
    lemmatized_text = []
    
    from pickle import dump
    lemmas = dict()
    with open ('./generate.txt', encoding='latin-1') as f:
        lines = f.readlines()
        
        lemmas = {}
        
        for line in lines:
            line = line.strip()
            if line != '':
                words = line.split()
                token = words[0].strip()
                token = token.replace('#', '')
                lemma = words[-1].strip()
                lemmas[token] = lemma
                
        print('\n There are {} tokens in the lemmas dictionary.'.format)
        lemmas_list = list(lemmas.items())
        print('\nThese are some entries in the lemmas dictionary\n []'.format)
    
def get_text(path):
    """Read text from a file, normalizing whitespace and stripping HTLM markup."""
    text = open(file= path, encoding='utf-8').read()
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    return text