# Author: Suárez Pérez Juan Pablo
# Date: 10/09/2022


# Import the libraries needed for the module...
from nltk.corpus import PlaintextCorpusReader
from bs4 import BeautifulSoup
from pickle import load
from pickle import dump
import nltk
import re

# Paths of importance
stopwords_path = 'C:/Users/USUARIO DELL/Documents/Python Scripts/PROCESAMIENTO_LENGUAJE_NATURAL/SIMILARITY_BM25/nlp_functions/stopwords_and_lemmas/stopwords_es.txt'
lemmas_path = 'C:/Users/USUARIO DELL/Documents/Python Scripts/PROCESAMIENTO_LENGUAJE_NATURAL/SIMILARITY_BM25/nlp_functions/stopwords_and_lemmas/generate.txt'



def clean_corpus(path_origin, path_destiny, no_page = 0):
    """
        Here, you can clean your corpus, so if you can get
        a text free HTML tags and save the corpus in an unique 
        arhive 'clean_corpus.txt'.
        path_origin: the path of your original corpus.
        path_destiny: the path of yout clean corpus.
    """
    corpus = PlaintextCorpusReader(path_origin, '.*')
    file_list = corpus.fileids()
    all_text = ''
    # Get all text of the corpus
    with open(path_origin + file_list[no_page], encoding = 'utf-8') as rfile:
        all_text = rfile.read()
    # Remove HTML tags
    soup = BeautifulSoup(all_text, 'lxml')
    clean_text = soup.get_text()
    # Apply the function lower to the text
    clean_text = clean_text.lower()
    # Save the file
    with open(path_destiny + 'clean_' + file_list[no_page][:11] + '_corpus.txt', 'w', encoding = 'utf-8') as file:
        file.write(clean_text)


def get_clean_text(path):
    """
        You can get the clean text without tags.
        path: the path of you clean text.
    """
    # Read file
    text = ''
    with open(path, encoding = 'utf-8') as file:
        text = file.read()
    return text


def word_tokenize(text):
    """
        Here you can tokenize yor clean corpus by words.
        text: the text you want to tokenize.
    """
    words = text.split()
    # Get only alphabetic words
    alphabetic_words = list()
    for word in words:
        token = list()
        for character in word:
            if re.match(r'^[a-záéíóúñü+$]', character):
                token.append(character)
        token = ''.join(token)
        if token != '':
            alphabetic_words.append(token)
    # Return tokens
    return alphabetic_words


def sentence_tokenize(text):
    """
        Here you can tokenize yor clean corpus by words.
        text: the text you want to tokenize.
    """
    tokens = nltk.data.load("tokenizers/punkt/spanish.pickle") 
    sents = tokens.tokenize(text)
    alphabetic_sents = list()
    for sent in sents:
        sent_token = word_tokenize(sent)        
        alphabetic_sents.append(sent_token)
    return alphabetic_sents


def delete_stop_words_sents(sents, path = stopwords_path):
    """
        Here you can delete the stop words from your sents.
        sents: the sents you want to clean.
        path: the path of you stopwords file.
    """
    with open(path, encoding = 'utf-8') as file:
        stop_words = file.readlines()
        stop_words = [w.strip() for w in stop_words]    
    clean_sents = list()
    for sent in sents:
        clean_sent = [word for word in sent if word not in stop_words]
        clean_sents.append(clean_sent)
    return clean_sents


def delete_stop_words(words, path = stopwords_path):
    """
        Here you can delete the stop words from your sents.
        sents: the sents you want to clean.
        path: the path of you stopwords file.
    """
    with open(path, encoding = 'utf-8') as file:
        stop_words = file.readlines()
        stop_words = [w.strip() for w in stop_words]    
    clean_words = [word for word in words if word not in stop_words]
    return clean_words


def lemmatize_sents(sents, path = lemmas_path):
    """
        Here you can lemmatize your sents.
        words: your sents free of stop words.
        path: the path where are the lemmas you want.
    """
    lemmas = dict()
    with open(path, encoding = 'latin-1') as file:
        lines = file.readlines()
        lines = [w.strip() for w in lines]
        for line in lines:
            line = line.strip()
            if line != '':
                words = line.split()
                token = words[0].strip()
                token = token.replace('#', '')
                lemma = words[-1].strip()
                lemmas[token] = lemma
    lemmas_sents = list()
    for sent in sents:
        lemmas_sent = list()
        for word in sent:
            if word in lemmas.keys():
                lemmas_sent.append(lemmas[word])
            else:
                lemmas_sent.append(word)
        lemmas_sents.append(lemmas_sent)
    return lemmas_sents


def lemmatize(text, path = lemmas_path):
    """
        Here you can lemmatize your sents.
        words: your sents free of stop words.
        path: the path where are the lemmas you want.
    """
    lemmas = dict()
    with open(path, encoding = 'latin-1') as file:
        lines = file.readlines()
        lines = [w.strip() for w in lines]
        for line in lines:
            line = line.strip()
            if line != '':
                words = line.split()
                token = words[0].strip()
                token = token.replace('#', '')
                lemma = words[-1].strip()
                lemmas[token] = lemma
    lemmas_word = list()
    for word in text:
        if word in lemmas.keys():
            lemmas_word.append(lemmas[word])
        else:
            lemmas_word.append(word)
    return lemmas_word


def get_vocabulary_by_sents(sents):
    """
        Here you can get the vocabulary of your sents.
        sents: list of sents.
    """
    words = list()
    for sent in sents:
        for word in sent:
            words.append(word)
    vocabulary = list(sorted(set(words)))
    print("Vocabulary size:", len(vocabulary))
    return vocabulary