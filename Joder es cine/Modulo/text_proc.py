from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import pandas as pd
from random import shuffle
import numpy as np
import os 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import nltk
import re
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XMLParser
from lxml import etree

def obtener_text(path = str(), file = str()):
    """
    Función que regresa el texto del archivo que se le pasa, preferentemente pasar la ruta relativa de donde se encuentra el archivo.
    Funciona con los archivos #.review.post del corpus del mismo directorio.
    Retorna el texto unicamente.
    path = ruta relativa o completa del archivo seleccionado.
    """
    
    with open(path + file, encoding='latin1', mode = 'r') as f:
        text = f.read()

    listas = text.split('\n')
    test = []
    
    for linea in listas:
        aux = linea.split(' ')
        try:
            test.append(aux[1]) 
        except:
            pass
    
    cad = ' '.join(test)
    return cad

def normalizar(text = str()):
    # nltk.download('stopwords') 
    '''
    Funcion para normalizar el texto y eliminar stopwords, así como signos de puntuación, guiones bajos y demás caracteres que no sean texto, retorna la cadena limpia.
    text : texto para normalizar
    '''
    stop_words = set(stopwords.words('spanish'))
    lower_string = text.lower()

    no_number_string = re.sub(r'\d+','',lower_string) 
    no_sub_ = re.sub('[\_]',' ', no_number_string)
    no_punc_string = re.sub(r'[^\w\s]','', no_sub_)  
    no_wspace_string = no_punc_string.strip() 
    # no_wspace_string 
    
    lst_string = [no_wspace_string][0].split() 
    # print(lst_string)
    no_stpwords_string="" 
    for i in lst_string: 
        if not i in stop_words: 
            no_stpwords_string += i+' '
            
    no_stpwords_string = no_stpwords_string[:-1]
    
    return no_stpwords_string

def get_rank (path = str(), file = str(), llave = 'rank'):
    """
    En la función solo se tiene que pasar el path, más el 
    archivo del cual se quiera obtener el rank, o mejor dicho 
    la valoración que se obtuvo en la pelicula, el archivo a pasar tiene que 
    ser en formato .xml para que la función funcione de forma correcta, 
    retorna el valor entero que se puso en la pelicula.
    
    path : ruta donde se encuentran los archivos xml
    file : nombre del archivo el cual se va a obtener el valor
    llave : atributo que se quiere, valor por defecto rank
    """
    
    with open(path + file, mode = 'r', encoding= 'latin1') as f:
        parser = etree.XMLParser(recover=True)
        tree = ET.parse(f, parser=parser)
        root = tree.getroot()
    
        att = root.attrib
    
    return int(att[llave])