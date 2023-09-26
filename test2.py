import nltk
from nltk.corpus import wordnet as wn


def max_similarity(word1, word2):
    """Finds the pair of definitions with the highest similarity between two words."""

    # Ensure the words are lowercased
    word1, word2 = word1.lower(), word2.lower()

    # Get the sets of synsets for each word
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 or not synsets2:
        raise ValueError("One of the words has no synsets.")

    # Initialize max similarity to be very low
    max_sim = 0
    max_pair = None

    # Compare each pair of synsets
    for syn1 in synsets1:
        for syn2 in synsets2:
            sim = syn1.path_similarity(syn2)
            # print(f"{syn1.definition()} - {syn2.definition()} - {sim}")
            if sim is not None and sim > max_sim:
                max_sim = sim
                max_pair = (syn1.definition(), syn2.definition())

    if max_pair is None:
        raise ValueError("No similarity could be computed.")

    return max_pair, max_sim


# # Example usage:
if __name__ == '__main__':
    word1 = "bright"
    word2 = "burn"
    (max_def1, max_def2), max_sim = max_similarity(word1, word2)
    print(f"Definitions with highest similarity:")
    print(f"{word1}: {max_def1}")
    print(f"{word2}: {max_def2}")
    print(f"Similarity: {max_sim}")

# import gensim.downloader as api
# from nltk.corpus import wordnet as wn
#
#
# def get_hypernyms(word):
#     """Get the hypernyms of a word."""
#     synsets = wn.synsets(word)
#     hypernyms = set()
#     for synset in synsets:
#         for hypernym in synset.hypernyms():
#             hypernyms.update(lemma.name() for lemma in hypernym.lemmas())
#     return list(hypernyms)
#
#
# def max_similarity_hypernyms(word1, word2, model):
#     """Finds the pair of hypernyms with the highest similarity between two words."""
#     hypernyms1 = get_hypernyms(word1)
#     hypernyms2 = get_hypernyms(word2)
#
#     max_sim = 0
#     max_pair = None
#
#     for hypernym1 in hypernyms1:
#         for hypernym2 in hypernyms2:
#             if hypernym1 in model and hypernym2 in model:
#                 sim = model.similarity(hypernym1, hypernym2)
#                 if sim > max_sim:
#                     max_sim = sim
#                     max_pair = (hypernym1, hypernym2)
#
#     return max_pair, max_sim
#
#
# # Load the Word2Vec model
# # model = api.load("word2vec-google-news-300")
# # Example usage:
# word1 = "ticks"
# word2 = "watch"
# for definition in wn.synsets(word2):
#     # print(definition.definition(), definition.lemma_names())
#     print("Definition: ", definition.definition(), "|", "Synonyms : ", definition.lemma_names())
# (max_hypernym1, max_hypernym2), max_sim = max_similarity_hypernyms(word1, word2, model)
# print(f"Hypernyms with highest similarity:")
# print(f"{word1} -> {max_hypernym1}")
# print(f"{word2} -> {max_hypernym2}")
# print(f"Similarity: {max_sim}")

# import nltk
# from nltk.corpus import wordnet as wn
#
#
# def get_synonyms(synset):
#     """Return a list of lemma names for a given synset and its hypernyms and hyponyms."""
#     synonyms = set()
#     for s in [synset] + synset.hyponyms() + synset.hypernyms():
#         for lemma in s.lemmas():
#             synonyms.add(lemma.name())
#     return synonyms
#
#
# def word_similarity(word1, word2):
#     """
#     Compute the maximum similarity between two words in WordNet.
#     """
#     synsets1 = wn.synsets(word1)
#     synsets2 = wn.synsets(word2)
#
#     max_sim = 0
#
#     # For each pair of synsets, calculate the similarity and update max_sim.
#     for syn1 in synsets1:
#         for syn2 in synsets2:
#             sim = syn1.wup_similarity(syn2)
#             if sim is not None and sim > max_sim:
#                 max_sim = sim
#
#     return max_sim
#
#
# def max_synonym_similarity(word1, word2):
#     """Finds the pair of synonyms with the highest similarity between two words."""
#
#     # Ensure the words are lowercased
#     word1, word2 = word1.lower(), word2.lower()
#
#     # Get the sets of synsets for each word
#     synsets1 = wn.synsets(word1)
#     synsets2 = wn.synsets(word2)
#
#     if not synsets1 or not synsets2:
#         raise ValueError("One of the words has no synsets.")
#
#     # Initialize max similarity to be very low
#     max_sim = 0
#     max_pair = None
#
#     # Compare each pair of synsets
#     for syn1 in synsets1:
#         for synonyms1 in get_synonyms(syn1):
#             for syn2 in synsets2:
#                 for synonyms2 in get_synonyms(syn2):
#                     sim = word_similarity(synonyms1, synonyms2)
#                     if sim > max_sim:
#                         max_sim = sim
#                         max_pair = (synonyms1, synonyms2)
#
#     if max_pair is None:
#         raise ValueError("No similarity could be computed.")
#
#     return max_pair, max_sim
#
#
# # Example usage:
# word1 = "ticks"
# word2 = "watch"
# (max_syn1, max_syn2), max_sim = max_synonym_similarity(word1, word2)
# print(f"Synonyms with highest similarity:")
# print(f"{word1}: {max_syn1}")
# print(f"{word2}: {max_syn2}")
# print(f"Similarity: {max_sim}")
