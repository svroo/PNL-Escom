# Author: Suárez Pérez Juan Pablo
# Date: 13/11/2022


import numpy as np

def get_most_freq_words(lemmatize_sents, vocabulary):
    tokens = dict()
    for word in vocabulary:
        tokens[word] = 0
    
    for sent in lemmatize_sents:
        for word in sent:
            tokens[word] += 1
    new_tokens = (sorted(tokens.items(), key = lambda item: item[1], reverse = True))
    with open('./most_freq_words/most_freq.txt', 'w', encoding = 'utf-8') as f:
        for item in new_tokens:
            f.write(str(item) + '\n')
    return tokens


def get_most_freq_words_prob(lemmatize_sents, vocabulary):
    tokens = dict()
    n = len(vocabulary)
    for word in vocabulary:
        tokens[word] = 0
    
    for sent in lemmatize_sents:
        for word in sent:
            tokens[word] += 1
    for k in tokens.keys():
        prob = tokens[k] /n
        tokens[k] = prob
    new_tokens = (sorted(tokens.items(), key = lambda item: item[1], reverse = True))
    with open('./most_freq_words/most_freq_prob.txt', 'w', encoding = 'utf-8') as f:
        for item in new_tokens:
            f.write(str(item) + '\n')
    return tokens


def get_most_freq_words_tf_idf(idf, freq, vocabulary):
    n = len(idf)
    tokens = dict()
    i = 0
    for word in vocabulary:
        tokens[word] = freq[word] * idf[i]
        i += 1
    tokens = (sorted(tokens.items(), key = lambda item: item[1], reverse = True))
    with open('./most_freq_words/most_freq_tf_idf.txt', 'w', encoding = 'utf-8') as f:
        for item in tokens:
            f.write(str(item) + '\n')
    return tokens


def get_distribution(titles, articles, topics):
    distribution_in_articles = dict()
    for n in range(len(titles)):
        distribution_in_articles[titles[n]] = get_freq(topics, articles[n])
    return distribution_in_articles


def get_freq(topics, article):
    freq = dict()
    for word in topics:
        freq[word] = 0
    for sent in article:
        for word in topics:
            if word in sent:
                freq[word] += 1
    pis = dict()
    sum = 0
    for v in freq.values():
        sum += v
    for k, v in freq.items():
        if sum != 0:
            pis[k] = v / sum
        else:
            pis[k] = 0
    return pis


def get_probs_word_background(articles, vocabulary):
    """
        Here you can get your Backgorund probabilities from your articles.
        articles: all your corpus.
        vocabulary: all the words that appear.
    """
    words =  list()
    probs_word_background = list()
    for art in articles:
        for sent in art:
            for word in sent:
                words.append(word)
    for v in vocabulary:
        probs_word_background.append(words.count(v) / len(vocabulary))
    probs_word_background = np.array(probs_word_background)
    return probs_word_background


def get_probs_word_topic(vocabulary):
    """
        Here you can get yout Topics probabilities from your articles.
        vocabulary: all the words that appear.
    """
    probs_word_topic = list()
    n = len(vocabulary)
    for v in vocabulary:
        probs_word_topic.append(1 / n)
    probs_word_topic = np.array(probs_word_topic)
    return probs_word_topic


def get_counts_article(article, vocabulary):
    """
        Here you can get the count for each word of the vocabulary in a given article.
        article: the article you want to get counts.
        vocabulary: list of unrepetead words of your corpus.
    """
    words = list()
    counts = list()
    for sentence in article:
        for word in sentence:
            words.append(word)
    for word in vocabulary:
        counts.append(words.count(word))
    counts = np.array(counts)
    return counts


def e_step(probs_word_background, probs_word_topic, prob_background = 0.5, prob_topic = 0.5):
    p_z_w = (prob_topic * probs_word_topic) / ((prob_topic * probs_word_topic) + (prob_background * probs_word_background))
    return p_z_w


def sum_cd(counts, e_step_i):
    sum_cd_i = 0
    n = len(counts)
    for i in range(n):
        sum_cd_i =  sum_cd_i + (counts[i] * e_step_i[i])
    return sum_cd_i


def m_step(counts, e_step_i):
    sum_cd_i = sum_cd(counts, e_step_i)
    new_p_w_t = (counts * e_step_i) / sum_cd_i
    return new_p_w_t


def log_maximum_likelihood(probs_word_topic, probs_word_background, counts, len_voc, prob_topic = 0.5, prob_background = 0.5):
    sum_max = 0
    for i in range(len_voc):
        sum_max = sum_max + (counts[i] * np.log((prob_background * probs_word_background[i]) + (prob_topic * probs_word_topic[i])))
    return sum_max


def em(probs_word_topic, probs_word_background, counts, len_voc, iterations = 50, prob_topic = 0.5, prob_background = 0.5):
    m_step_i = probs_word_topic
    for i in range(iterations):
        probs_word_topic_n = m_step_i
        e_step_i = e_step(probs_word_background, probs_word_topic_n, prob_background, prob_topic)
        log_maximum_likelihood_i = log_maximum_likelihood(probs_word_topic_n, probs_word_background, counts, len_voc, prob_topic, prob_background)
        m_step_i = m_step(counts, e_step_i)
        print("Iteration", i, ":")
        print("\tLog Maximum Likelihood:", log_maximum_likelihood_i)
    return m_step_i


def create_dict(list_keys, list_values):
    new_dict = dict(zip(list_keys, list_values))
    return new_dict


def sort_dict(old_dict, keys = False, reverse_value = True):
    if keys:
        new_dict = sorted(old_dict.items(), key = lambda item: item[0], reverse = reverse_value)
    else:
        new_dict = sorted(old_dict.items(), key = lambda item: item[1], reverse = reverse_value)
    return new_dict