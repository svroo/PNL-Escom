# Import the libraries needed for the module...
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import cess_esp
from bs4 import BeautifulSoup
from pickle import load
from pickle import dump
import re
import nltk

# Paths of importance
stopwords_path = 'C:\\Users\\hp\\Documents\\VSCode\\PNL-Escom\\Primer parcial\\Ejercicios en clase\\stopwords_es.txt'
lemmas_path = 'C:\\Users\\hp\\Documents\VSCode\\PNL-Escom\\Primer parcial\\Ejercicios en clase\\generate.txt'
spanish_tagger = 'C:\\Users\\hp\\Documents\\VSCode\\PNL-Escom\\Primer parcial\\Ejercicios en clase\\tag\\spanish_tagger.pkl'

def clean_corpus(path_origin, path_destiny):
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
    for file in file_list:
        with open(path_origin + file, encoding = 'utf-8') as rfile:
            text = rfile.read()
            all_text += text
    # Remove HTML tags
    soup = BeautifulSoup(all_text, 'lxml')
    clean_text = soup.get_text()
    # Apply the function lower to the text
    clean_text = clean_text.lower()
    # Save the file
    with open(path_destiny + 'clean_corpus.txt', 'w', encoding = 'utf-8') as file:
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

# Modificada
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


def delete_stop_words(words, path = stopwords_path):
    """
        Here you can delete the stop words.
        words: the words you want to clean.
        path: the path of you stopwords file.
    """
    # Read the stop words
    with open(path, encoding = 'utf-8') as file:
        stop_words = file.readlines()
        stop_words = [w.strip() for w in stop_words]
    clean_words = [word for word in words if word not in stop_words]
    return clean_words

# Modificada
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


def lemmatize(text, path = lemmas_path):
    """
        Here you can lemmatize your words.
        words: your words free of stop words.
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
    lemmatized_text = list()
    for word in text:
        if word in lemmas.keys():
            lemmatized_text.append(lemmas[word])
        else:
            lemmatized_text.append(word)
    return lemmatized_text

# Modificada
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
                tag = words[-2].strip()
                tag = tag[0].lower()
                lemmas[(token, tag)] = (lemma, tag)
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


def get_vocabulary(words):
    """
        Here you can get the vocabulary of your words.
        words: list of words.
    """
    vocabulary = list(sorted(set(words)))
    return vocabulary

# Nueva
def make_and_save_spanish_tagger(fname):

    tags_sents = list()
    for sent in cess_esp.tagged_sents():
        tags_sents_aux = [tag for (word, tag) in sent]
        tags_sents = tags_sents + tags_sents_aux

    most_used_tag_sents = nltk.FreqDist(tags_sents).max()

    default_tagger = nltk.DefaultTagger(most_used_tag_sents)

    patterns = [
        (r'.o$', 'n'),
        (r'.os$', 'n'),
        (r'.a$', 'n'),
        (r'.as$', 'n'),
        (r'.e$', 'n'),
        (r'.es$', 'n'),
        (r'.^[0-9]+$', 'z')
    ]

    regexp_tagger = nltk.RegexpTagger(patterns, backoff = default_tagger)

    cess_tagged_sents = cess_esp.tagged_sents()
    spanish_tagger = nltk.UnigramTagger(cess_tagged_sents, backoff = regexp_tagger)

    output = open(fname, 'wb')
    dump(spanish_tagger, output, -1)
    output.close() 
    
# Nueva    
def tag(sents, path = spanish_tagger):
    input_f = open(path, 'rb')
    tagger = load(input_f)
    input_f.close()
    tagged_sentences = [tagger.tag(sent) for sent in sents]
    return tagged_sentences