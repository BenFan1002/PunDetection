import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np

# Load the Word2Vec model (this could be quite large)
model = api.load("word2vec-google-news-300")


def potential_pun(word, context, threshold=0.6):
    # Find the most similar words to the given word using Word2Vec
    similar_words = [item[0] for item in model.most_similar(word, topn=10)]

    # For each similar word, check its similarity with the context
    context_vector = np.mean([model[token] for token in context if token in model.key_to_index], axis=0)
    scores = [model.similarity(word, token) for token in similar_words]

    # If there are multiple meanings supported by the context, the word might be a pun
    high_score_count = sum(1 for score in scores if score > threshold)
    return high_score_count < 1


def detect_pun(sentence):
    tokens = word_tokenize(sentence)
    puns = [word for word in tokens if word in word in model.key_to_index and potential_pun(word, tokens)]
    return puns


sentence = "I'm reading a book on anti-gravity. It's impossible to put down!"
print(detect_pun(sentence))
