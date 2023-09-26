import math
import string
import gensim.downloader as api
import torch
from nltk import word_tokenize, MWETokenizer
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForMaskedLM
from nltk.corpus import wordnet as wn

stop_words = set(stopwords.words('english'))


# model = api.load("word2vec-google-news-300")


def get_definitions(word):
    synsets = wn.synsets(word)
    definitions = [syn.definition() for syn in synsets]
    return definitions


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


def mask_and_predict(sentence, top_n=5):
    # Load pre-trained model and tokenizer
    model_name = 'bert-base-uncased'
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()

    words = sentence.split()
    results = []

    # Loop through each word in the sentence
    for i in range(len(words)):
        if words[i] in string.punctuation:
            continue
        if words[i] in stop_words:
            continue
        masked_sentence = ' '.join([words[j] if j != i else '[MASK]' for j in range(len(words))])
        inputs = tokenizer(masked_sentence, return_tensors="pt")

        # Get model output
        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits

        # Get top_n predictions for the masked word
        masked_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()
        probs = logits[0, masked_index].softmax(dim=0)
        top_prob_values, top_indices = torch.topk(probs, top_n)
        # print(f"Original Word: {words[i]}")
        # for idx, token_id in enumerate(top_indices):
        #     predicted_token = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
        #     prob_value = top_prob_values[idx].item()
        #     print(f"    Candidate {idx + 1}: {predicted_token} with probability {prob_value:.4f}")

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


def sigmoid(x):
    """Returns the sigmoid of x."""
    return 1 / (1 + math.exp(-x))


def definition_avg_similarity(definition, context_word):
    """
    Compute the average similarity of words in the definition to the context word.

    :param definition: The definition text.
    :param context_word: The context word to compare against.
    :param model: Preloaded Word2Vec model.
    """
    tokens = [word for word in definition.lower().split() if word.isalnum()]

    total_similarity = 0
    valid_token_count = 0
    for token in tokens:
        if token in model:
            similarity = model.similarity(context_word, token)
            total_similarity += similarity
            valid_token_count += 1

    avg_similarity = total_similarity / valid_token_count if valid_token_count else 0

    return avg_similarity


def context_relevant_definition(word, context_word):
    word_synsets = wn.synsets(word)
    if not word_synsets:
        return None

    best_definition = None
    max_avg_similarity = -1
    for w_syn in word_synsets:
        current_avg_similarity = definition_avg_similarity(w_syn.definition(), context_word)
        print(f"{context_word} Definition: '{w_syn.definition()}' has average similarity: {current_avg_similarity}")
        if current_avg_similarity > max_avg_similarity:
            max_avg_similarity = current_avg_similarity
            best_definition = w_syn.definition()

    return best_definition


def main(sentence="OLD GEOGRAPHERS never die , they just become legends ."):
    tokenizer = MWETokenizer()  # Multi-word expression tokenizer
    tokens = tokenizer.tokenize(word_tokenize(sentence))
    # definition = context_relevant_definition(word, context_words)

    predictions = [x for x in mask_and_predict(" ".join(tokens))]
    print(predictions)
    predictions.sort(key=lambda x: x[2], reverse=False)
    filtered_data = predictions[:3]
    print(filtered_data)
    if filtered_data[0][2] < 0.2:
        print(f"Sentence to detect pun: {sentence}")
        # find element the lowest probability
        pun_word = min(filtered_data, key=lambda x: x[2])
        if len(wn.synsets(pun_word[0])) <2:
            pun_word = filtered_data[1]
        filtered_data.remove(pun_word)
        # pun_word = pun_word[0]
        print(f"This is a pun! The word '{pun_word[0]}' has multiple meanings.")
        if len(wn.synsets(pun_word[0])) == 2:
            for index, definition in enumerate(wn.synsets(pun_word[0])):
                print(f"Definition {index + 1}: {definition.definition()}. ", end="")
            return
        # for index, item in enumerate(filtered_data):
        #     print(f"Definition {index + 1}: {context_relevant_definition(pun_word, item[0])}. ", end="")
        from test2 import max_similarity
        for index, item in enumerate(filtered_data):
            # print(max_similarity(pun_word[0], item[0]))
            print(f"Definition {index + 1}: {max_similarity(pun_word[0], item[0])[0][0]}. ", end="")
    else:
        print("This is not a pun!")


if __name__ == '__main__':
    main("""Danny: Why so glum?
Gerry: I've got a bad case of shingles.
Danny: Did you see a doctor?
Gerry: Yeah, and he prescribed aluminum siding. """)
