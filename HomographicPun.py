import string

import nltk
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertForMaskedLM

from Get_Definition import word2VecModel, find_most_similar_definition, find_most_similar_definition_homographic

stop_words = set(stopwords.words('english'))


# model = api.load("word2vec-google-news-300")

def get_definitions(word):
    synsets = wn.synsets(word)
    definitions = [syn.definition() for syn in synsets]
    return definitions


def convert_to_nltk_postag(postag):
    pos_tag_mappings = {
        'noun': wn.NOUN,
        'verb': wn.VERB,
        'adj': wn.ADJ,
        'adv': wn.ADV
    }
    return pos_tag_mappings.get(postag.lower(), None)


def get_word_definitions(word, pos_tag):
    nltk_postag = convert_to_nltk_postag(pos_tag)

    # Find synsets considering the part of speech
    synsets = wn.synsets(word, pos=nltk_postag)
    return synsets


def context_relevant_definition(word, context_word):
    context_synsets = wn.synsets(context_word)
    word_synsets = wn.synsets(word)

    # If either word has no synsets, return None
    if not context_synsets or not word_synsets:
        return None

    best_definition = None
    max_similarity = -1

    for w_syn in word_synsets:
        for c_syn in context_synsets:
            similarity = wn.wup_similarity(w_syn, c_syn)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
                best_definition = w_syn.definition()

    return best_definition, max_similarity  # Return tuple with definition and similarity


model_name = 'bert-base-uncased'
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()


def mask_and_predict(sentence, top_n=5):
    # Load pre-trained model and tokenizer

    words = sentence
    results = []

    # Loop through each word in the sentence
    for i in range(len(words)):
        word = words[i][0]
        if word in string.punctuation:
            continue
        if word in stop_words:
            continue
        masked_sentence = ' '.join([words[j][0] if j != i else '[MASK]' for j in range(len(words))])
        inputs = tokenizer(masked_sentence, return_tensors="pt")

        # Get model output
        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits

        # Get top_n predictions for the masked word
        masked_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()
        probs = logits[0, masked_index].softmax(dim=0)
        top_prob_values, top_indices = torch.topk(probs, top_n)
        top_prediction = tokenizer.convert_ids_to_tokens([top_indices[0].item()])[0]
        results.append((words[i], top_prediction, top_prob_values[0].item()))

    return results


def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))


def compute_similarity(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    # if there are no synsets, return 0
    if not synsets1 or not synsets2:
        return 0

    best_score = 0
    # iterate over pairs of synsets and compute their path similarity
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity is not None and similarity > best_score:
                best_score = similarity

    return best_score


def max_similarity(word1, word2):
    """Finds the pair of definitions with the highest similarity between two words."""
    if word1[0].capitalize() not in word2VecModel or word2[0].replace("#", '').capitalize() not in word2VecModel:
        return None
    return word2VecModel.similarity(word1[0].capitalize(), word2[0].replace("#", '').capitalize())


def getSimilarity(word):
    if word[0][0].capitalize() not in word2VecModel or word[1].capitalize() not in word2VecModel:
        return 1
    return word2VecModel.similarity(word[0][0].capitalize(), word[1].replace('#', '').capitalize())


def main(sentence="OLD GEOGRAPHERS never die, they just become legends."):
    # Tokenize and tag the parts of speech in the sentence
    pos_tagged_words = nltk.pos_tag(word_tokenize(sentence))

    # Get predictions for possible pun words
    pun_predictions = [x for x in mask_and_predict(pos_tagged_words)]
    # pun_predictions.sort(key=getSimilarity, reverse=False)
    pun_predictions.sort(key=lambda x: x[2], reverse=False)

    # Filter predictions to those that have definitions
    valid_predictions = [[(word, pos_tag), predictor, score] for (word, pos_tag), predictor, score in pun_predictions if
                         get_word_definitions(word, pos_tag)]
    top_predictions = valid_predictions[:3]

    if len(top_predictions) < 3:
        return {
            'is_Homographic_Pun': False,
            'Reason': 'Not enough data to determine a pun.'
        }
    possible_pun_words = (word for word in top_predictions if len(get_word_definitions(*(word[0]))) > 1)
    # Find the word with multiple meanings
    while True:
        potential_pun_word = next(possible_pun_words, None)

        if potential_pun_word is None:
            # print("No potential pun word found.")
            return {
                'is_Homographic_Pun': False,
                'Reason': 'No potential pun word found'
            }
        if potential_pun_word[0][0] not in word2VecModel or potential_pun_word[1].replace("#", '').capitalize() not in word2VecModel:
            continue
        # Calculate similarity for the potential pun word
        # pun_word_similarity = max_similarity(potential_pun_word[0], (potential_pun_word[1], potential_pun_word[0][1]))
        #
        # if pun_word_similarity < 0.15:
        dictionary = find_most_similar_definition_homographic([sentence, potential_pun_word[0][0]],
                                                              potential_pun_word[0][0], potential_pun_word[0][1])
        if not dictionary:
            continue
        if dictionary['two definitions similarity'] > 0.95:
            continue
        first_def, first_sim = dictionary['first definition']['definition'], dictionary['first definition'][
            'similarity']
        second_def, second_sim = dictionary['second definition']['definition'], \
            dictionary['second definition']['similarity']
        return {
            'is_Homographic_Pun': True,
            'pun_word': potential_pun_word[0][0],
            'first_definition': first_def,
            'first_similarity': first_sim,
            'second_definition': second_def,
            'second_similarity': second_sim,
        }


if __name__ == '__main__':
    print(
        main("He put bug spray on his watch to get rid of the ticks .")
    )
