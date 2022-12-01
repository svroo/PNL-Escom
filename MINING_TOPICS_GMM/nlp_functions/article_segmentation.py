# Author: Suárez Pérez Juan Pablo
# Date: 13/11/2022


# Import the libraries needed for the module...
from nltk.corpus import PlaintextCorpusReader
from bs4 import BeautifulSoup
from pickle import load
from pickle import dump
import numpy as np

# Paths of importance
stopwords_path = 'C:/Users/USUARIO DELL/Documents/Python Scripts/PROCESAMIENTO_LENGUAJE_NATURAL/SIMILARITY_BM25/nlp_functions/stopwords_and_lemmas/stopwords_es.txt'
lemmas_path = 'C:/Users/USUARIO DELL/Documents/Python Scripts/PROCESAMIENTO_LENGUAJE_NATURAL/SIMILARITY_BM25/nlp_functions/stopwords_and_lemmas/generate.txt'


def is_title(text):
    lower = 0
    upper = 0
    for c in text:
        if c.isalpha():
            if c.isupper():
                upper += 1
            else:
                lower += 1
    if lower > upper:
        return True
    return False


def get_titles(path_origin, path_destiny, no_page = 0):
    """
        Here you can get the titles of your text.
        path_origin: the path of your orginal corpus.
        no_pages: the number of pages you want to consider. 
    """
    corpus = PlaintextCorpusReader(path_origin, '.*')
    file_list = corpus.fileids()
    titles = list()
    with open(path_origin + file_list[no_page], encoding = 'utf-8') as rfile:
        text = rfile.read()
    soup = BeautifulSoup(text, 'lxml')
    sents = soup.find_all('h3')
    for sent in sents:
        txt = sent.get_text()
        if is_title(txt):
            if txt != ' ':
                titles.append(txt.lower())
    with open(path_destiny + "titles_" + file_list[no_page][:11] + '.txt', 'w', encoding = 'utf-8') as wfile:
        wfile.writelines(titles)
    return titles


def get_subtitles(path_origin, path_destiny, no_page = 0):
    """
        Here you can get the subtitles of your text.
        path_origin: the path of your orginal corpus.
        no_pages: the number of pages you want to consider. 
    """
    corpus = PlaintextCorpusReader(path_origin, '.*')
    file_list = corpus.fileids()
    subtitles = list()
    with open(path_origin + file_list[no_page], encoding = 'utf-8') as rfile:
        text = rfile.read()
    soup = BeautifulSoup(text, 'lxml')
    sents = soup.find_all(['b'])
    for sent in sents:
        txt = sent.get_text()
        if is_title(txt):
            if txt != ' ':
                subtitles.append(txt.lower())
    new_subtitles = [subtitle + '\n' for subtitle in subtitles]
    with open(path_destiny + "subtitles_" + file_list[no_page][:11] + '.txt', 'w', encoding = 'utf-8') as wfile:
        wfile.writelines(new_subtitles)
    return subtitles


def get_articles(path_origin, no_page = 0):
    """
        Here you can get the articles of your text.
        path_origin: the path of your orginal corpus.
        no_pages: the number of pages you want to consider. 
    """
    corpus = PlaintextCorpusReader(path_origin, '.*')
    file_list = corpus.fileids()
    with open(path_origin + file_list[no_page], encoding = 'utf-8') as rfile:
        text = rfile.read()
    arts = text.split("\n\n\n")[1:]
    new_arts = arts[1::2]
    clean_arts = list()
    for art in new_arts:
        clean_arts.append(clean_articles(art))
    return clean_arts


def clean_articles(txt):
    """
        Here you can remove all yout HTML tags from your articles.
        txt: the article you want to clean.
    """
    soup = BeautifulSoup(txt, 'lxml')
    clean_text = soup.get_text()
    # Apply the function lower to the text
    clean_text = clean_text.lower()
    # Save the file
    return clean_text


def get_vocabulary_from_articles(articles):
    words = list()
    for article in articles:
        for sent in article:
            for w in sent:
                words.append(w)
    vocabulary = list(sorted(set(words)))
    return vocabulary