from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet as wn
from torch.nn import functional as F

word2VecModel = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def construct_sentences_from_definitions(word):
    sentences = []
    for synset in wn.synsets(word):
        # Get the word's part of speech
        pos = synset.pos()
        # Capitalize the word and get the definition
        definition = synset.definition()
        # Determine the article to use ('a' or 'an')
        # article = 'an' if pos[0] in 'aeiou' else 'a'
        # Construct the sentence based on the part of speech
        if pos == 'n':
            sentence = f"A {word} is {definition}."
        elif pos == 'v':
            sentence = f"To {word} is to {definition}."
        elif pos == 'a' or pos == 's':  # Adjective and adjective satellite
            sentence = f"{word.capitalize()} is {definition}."
        elif pos == 'r':  # Adverb
            sentence = f"{word.capitalize()} is describes {definition}."
        else:
            sentence = f"{word.capitalize()} definition: {definition}."

        sentences.append([sentence, word])
    return sentences


def construct_sentences_from_definitions_homographic(word, pos_tag):
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    sentences = []
    for synset in wn.synsets(word):
        # Get the word's part of speech
        pos = synset.pos()
        # Capitalize the word and get the definition
        definition = synset.definition()
        # Determine the article to use ('a' or 'an')
        # article = 'an' if pos[0] in 'aeiou' else 'a'
        # Construct the sentence based on the part of speech
        if get_wordnet_pos(pos_tag) and pos != get_wordnet_pos(pos_tag):
            continue
        if pos == 'n':
            sentence = f"A {word} is {definition}."
        elif pos == 'v':
            sentence = f"To {word} is to {definition}."
        elif pos == 'a' or pos == 's':  # Adjective and adjective satellite
            sentence = f"{word.capitalize()} is {definition}."
        elif pos == 'r':  # Adverb
            sentence = f"{word.capitalize()} is describes {definition}."
        else:
            sentence = f"{word.capitalize()} definition: {definition}."

        sentences.append([sentence, word])
    return sentences


def cosine_similarity_percentage(vector1, vector2):
    """
    Compute the cosine similarity between two vectors and convert it to a percentage.

    :param vector1: A PyTorch tensor representing the first vector.
    :param vector2: A PyTorch tensor representing the second vector.
    :return: A float value representing the cosine similarity percentage.
    """
    # Ensure the vectors are 2D
    vector1 = vector1.unsqueeze(0) if vector1.dim() == 1 else vector1
    vector2 = vector2.unsqueeze(0) if vector2.dim() == 1 else vector2

    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(vector1, vector2, dim=1)

    # Convert to percentage
    similarity_percentage = (cosine_sim + 1) / 2 * 100

    # Extract the single similarity value as a Python float
    return similarity_percentage.item()


def get_sense_vector(sentenceList):
    sentence, word = sentenceList
    # Tokenize the sentence and add the special tokens
    inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)

    # Get the input IDs and attention mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Pass the inputs to the model
    outputs = model(input_ids, attention_mask=attention_mask)

    # Get the last hidden states
    last_hidden_states = outputs.last_hidden_state

    # Tokenize the word and get its token IDs
    word_tokens = tokenizer.tokenize(word)
    word_ids = tokenizer.convert_tokens_to_ids(word_tokens)

    # Find the positions of the word's tokens in the input IDs
    token_positions = [i for i, input_id in enumerate(input_ids[0]) if input_id in word_ids]

    # Extract the hidden states of the word's tokens
    word_hidden_states = last_hidden_states[0, token_positions, :]

    # Calculate the mean of the hidden states for the word's tokens
    sense_vector = word_hidden_states.mean(dim=0)

    return sense_vector


def find_most_similar_definition(sentence, similar_words):
    """
    This function takes a word, a sentence, and a dictionary of similar words.
    It finds the WordNet definitions for the original word and a similar sounding word,
    computes the cosine similarity with the sentence context, and returns the most similar definitions.

    Parameters:
    word (str): The word to find definitions for.
    sentence (list): The context sentence in which the word is used.
    similar_words (dict): A dictionary with keys 'original word' and 'similar sounding'
                          containing words to compare definitions for.

    Returns:
    tuple: A tuple containing the most similar definition sentence and its similarity percentage for each word.
    """

    # Get the sense vector for the original word in the context of the provided sentence
    original_vector = get_sense_vector(sentence)

    # Get sense vectors for each definition of the original word from WordNet
    original_definition_vectors = []
    for sentence in construct_sentences_from_definitions(similar_words['original word']):
        vector = get_sense_vector(sentence)
        original_definition_vectors.append((vector, sentence))

    # Compute cosine similarity between the original sentence vector and each original word definition vector
    original_similarities = []
    for vector, def_sentence in original_definition_vectors:
        similarity = cosine_similarity_percentage(original_vector.unsqueeze(0), vector.unsqueeze(0))
        original_similarities.append((similarity, def_sentence))

    # Find the definition with the highest similarity score for the original word
    most_similar_original = max(original_similarities, key=lambda x: x[0])

    # Replace the original word with the similar sounding word in the sentence
    sentence_with_similar = [s.replace(similar_words['original word'], similar_words['similar sounding']) for s in
                             sentence]

    # Get the sense vector for the similar sounding word in the context of the modified sentence
    similar_vector = get_sense_vector(sentence_with_similar)

    # Get sense vectors for each definition of the similar sounding word from WordNet
    similar_definition_vectors = []
    for sentence in construct_sentences_from_definitions(similar_words['similar sounding']):
        vector = get_sense_vector(sentence)
        similar_definition_vectors.append((vector, sentence))
    # Compute cosine similarity between the sentence vector with the similar word and each similar word definition
    # vector
    similar_similarities = []
    for vector, def_sentence in similar_definition_vectors:
        similarity = cosine_similarity_percentage(similar_vector.unsqueeze(0), vector.unsqueeze(0))
        similar_similarities.append((similarity, def_sentence))

    # Find the definition with the highest similarity score for the similar sounding word
    most_similar_similar = max(similar_similarities, key=lambda x: x[0])

    return {
        'original word': {
            'definition': most_similar_original[1][0],
            'similarity': most_similar_original[0]
        },
        'similar sounding': {
            'definition': most_similar_similar[1][0],
            'similarity': most_similar_similar[0]
        }
    }


def find_most_similar_definition_homographic(sentence, similar_word, pos_tag):
    # Get the sense vector for the original word in the context of the provided sentence
    original_vector = get_sense_vector(sentence)

    # Get sense vectors for each definition of the original word from WordNet
    original_definition_vectors = []
    for s in construct_sentences_from_definitions_homographic(similar_word, pos_tag):
        vector = get_sense_vector(s)
        original_definition_vectors.append((vector, s))

    # Compute cosine similarity between the original sentence vector and each original word definition vector
    original_similarities = []
    for vector, def_sentence in original_definition_vectors:
        similarity = cosine_similarity_percentage(original_vector.unsqueeze(0), vector.unsqueeze(0))
        original_similarities.append((similarity, def_sentence))
    if len(original_similarities) < 2:
        return None
    # Find the definition with the highest similarity score for the original word
    original_similarities.sort(key=lambda x: x[0], reverse=True)
    two_definitions_similarity = cosine_similarity_percentage(get_sense_vector(original_similarities[0][1]),
                                                              get_sense_vector(original_similarities[1][1]))
    return {
        'first definition': {
            'definition': original_similarities[0][1][0],
            'similarity': original_similarities[0][0]
        },
        'second definition': {
            'definition': original_similarities[1][1][0],
            'similarity': original_similarities[1][0]
        },
        'two definitions similarity': two_definitions_similarity
    }


if __name__ == '__main__':
    # results = find_most_similar_definition(
    #     ["People who like gold have guilt complex.", 'guilt'],
    #     {'keyword': 'Gold', 'similar sounding': 'gilt', 'original word': 'guilt'})
    # print(results)
    print(find_most_similar_definition_homographic(["He put bug spray on his watch to get rid of the ticks .", 'ticks'],
                                                   'ticks', 'NNS'))
