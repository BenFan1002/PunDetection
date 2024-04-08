import math
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import string
import re
from weighted_levenshtein import lev
from Get_Definition import word2VecModel, tokenizer, model, find_most_similar_definition

punctuation_without_apostrophe = string.punctuation.replace("'", "")

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


def clean_and_compare(word1, word2):
    # Define a regex pattern to find parentheses with numbers inside
    parentheses_with_numbers = re.compile(r'\(\d+\)')

    # Remove parentheses with numbers and commas, then convert to uppercase
    clean_word1 = re.sub(parentheses_with_numbers, '', word1).replace('\'', '').upper()
    clean_word2 = re.sub(parentheses_with_numbers, '', word2).replace('\'', '').upper()

    # Compare the cleaned words
    return clean_word1 == clean_word2


def getCosts():
    import numpy as np
    import pandas as pd

    # Initialize the substitute_costs matrix
    substitute_costs = np.ones((128, 128), dtype=np.float64)
    delete_costs = np.ones(128, dtype=np.float64)
    insert_costs = np.ones(128, dtype=np.float64)
    df = pd.read_excel('Prouncation Table.xlsx', header=0, index_col=0)

    # Iterate through the DataFrame to update the substitute_costs

    for row_key in df.index:
        if row_key == ".":  # Skip the space row
            continue
        for col_key in df.columns:
            # Convert the row and column labels to ASCII if they are single characters
            row_index = ord(str(row_key))
            col_index = ord(str(col_key))
            if col_key == ".":
                delete_costs[row_index] = df.at[row_key, col_key]
                insert_costs[col_index] = df.at[row_key, col_key]
                continue
            # Update the substitute_costs matrix
            substitute_costs[row_index, col_index] = df.at[row_key, col_key]
    return substitute_costs, delete_costs, insert_costs


substitute_costs, delete_costs, insert_costs = getCosts()


def cmu_to_paper(word):
    paper_variants = {
        "AE0": "0",
        "AE1": "&",
        "AE2": "&",
        "OY0": "3",
        "OY2": "3",  # Not sure about this
        "OY1": "3",
        "T": "?",  # Sometimes t, sometimes ?
        "L": "l",
        "AY0": "1",
        "AY1": "1",
        "AY2": "1",
        "S": "s",
        "W": "w",
        "D": "d",
        "AW0": "2",
        "AW1": "2",
        "AW2": "2",
        "EH": "7",
        "EH0": "0",
        "EH2": "E",
        "EH1": "E",
        "TH": "T",
        "ZH": "Z",
        "R": "r",
        "M": "m",
        "K": "k",
        "CH": "C",
        "EY2": "4",
        "EY1": "4",
        "EY0": "4",
        "G": "g",
        "IY0": "i",
        "IY1": "i",
        "IY2": "i",
        "AO0": "O",
        "AO1": "O",
        "AO2": "O",
        "IH0": "I",
        "IH1": "I",  # dinner should be I
        "IH2": "0",
        "SH": "S",
        "UW0": "u",
        "UW1": "u",
        "UW2": "u",  # Not sure
        "NG": "N",
        "B": "b",
        "P": "p",
        "AA": "a",
        "AA0": "a",
        "AA1": "a",
        "AA2": "0",
        "HH": "h",
        "AH0": "I",  # Evil should be I, decimal should be 0
        "AH1": "I",
        "AH2": "^",
        "OW0": "5",
        "OW1": "5",
        "OW2": "5",
        "Z": "z",
        "V": "v",
        "N": "n",
        "DH": "D",
        "JH": "J",
        "Y": "j",
        "F": "f",
        "ER1": "9r",
        "ER0": "0r",
        "ER2": "0r",
        "UH": "8",
        "UH0": "8",
        "UH2": "8",  # Not sure about this one
        "UH1": "U",
    }

    # Split the word into phonemes
    phonemes = word.strip().split(" ")

    # Convert CMU phonemes to paper phonemes
    paper_sound = []
    for phoneme in phonemes:
        cmu_phoneme = phoneme.upper()
        paper_phoneme = paper_variants[cmu_phoneme]
        paper_sound.append(paper_phoneme)

    # Join the converted phonemes back into a single string
    return "".join(paper_sound)


stop_words = set(stopwords.words('english'))
stop_words.add("who's")
stop_words.add("sure")
stop_words.add("can't")
stop_words.add("cant")
stop_words.remove("too")


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', punctuation_without_apostrophe))


def extract_letters(input_string):
    # This regex will match any sequence of letters and ignore any non-letter characters.
    matches = re.findall(r'[a-zA-Z]+', input_string)
    # If you want to concatenate all matches into one string of letters:
    all_letters = ''.join(matches)
    return all_letters


def find_similar_sounding_words(input_word, cmu_dict, num_results=10):
    # Normalize the input word to uppercase as CMUdict is in uppercase
    input_word = input_word.upper()

    # Get the phonetic transcription(s) for the input word
    if " " in input_word:
        # If the input word contains spaces, split it into words and get the transcription for each word
        transcription = []
        for word in input_word.split(" "):
            transcription.append(cmu_dict[word])
        transcription = " ".join(transcription)
    else:
        # If the input word is a single word, get the transcription
        transcription = cmu_dict.get(input_word, None)
    # If the word is not in the dictionary, return a message
    if not transcription:
        return None

    # Calculate the edit distance for each transcription of the input word
    # against all other words in the dictionary
    similarity_scores = []
    for word, word_transcription in cmu_dict.items():
        # word_transcription = word_transcription.split(" ")
        # if len(word_transcription) != len(transcription):
        #     continue
        if not clean_and_compare(word, input_word):
            try:
                score = lev(cmu_to_paper(transcription), cmu_to_paper(word_transcription),
                            # insert_costs=insert_costs,
                            # delete_costs=delete_costs,
                            substitute_costs=substitute_costs)
            except Exception:
                print(word, input_word)
            # if word == "sale".upper():
            #     print("transcription: ", f"{transcription}", "word_transcription: ", f"{word} {word_transcription}",
            #           "score: ", score)
            similarity_scores.append((score, word.capitalize()))

    # Sort the words by the edit distance and get the top results
    similarity_scores.sort()
    # similar_words = [word for score, word in similarity_scores[:num_results]]

    return similarity_scores


cmu_dict = {}

# Trying to open the file with a different encoding if UTF-8 fails
with open("cmudict-0.7b", 'r', encoding='ISO-8859-1') as file:
    for line in file:
        if not line.startswith(";;;"):  # Skip comment lines
            parts = line.strip().split('  ')
            if len(parts) == 2:
                word, transcription = parts
                cmu_dict[word] = transcription


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(0.5 * x)) + (0.1 if x == 0 else 0)


def detect_pun(sentence):
    cleanedSentence = remove_punctuation(
        sentence.replace("’", "'")
    ).replace("Whos", "Who's").replace('Im', 'I\'m').split()

    similar_sounding_words = []

    # Find homophones for each word
    for word in cleanedSentence:
        if word.lower() in stop_words:
            continue

        similar_words = find_similar_sounding_words(word, cmu_dict)
        if not similar_words:
            continue
        for score, homophone in similar_words:
            if score < 0.3 and homophone.capitalize() in word2VecModel and homophone.lower() in word2VecModel:
                similar_sounding_words.append((word, homophone, score))

    if similar_sounding_words:
        max_similarity = 0.05
        most_similar_word = None
        # For each homophone, find the word in the sentence with the highest similarity
        for word, homophone, score in similar_sounding_words:
            for other_word in cleanedSentence:
                if homophone.capitalize() not in word2VecModel or other_word.capitalize() not in word2VecModel:
                    continue
                if other_word.lower() not in stop_words and other_word != word:
                    # Calculate similarity with sigmoid adjustment
                    similarity = (
                            word2VecModel.similarity(homophone.capitalize(), other_word.capitalize()) * sigmoid(score)
                    )
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_word = {
                            "keyword": other_word.capitalize(),
                            "similar sounding": homophone,
                            "original word": word,
                        }
        if not construct_sentences_from_definitions(most_similar_word['similar sounding']):
            return {
                "is_Hetergraphic_Pun": False,
            }
        if not construct_sentences_from_definitions(most_similar_word['original word']):
            return {
                "is_Hetergraphic_Pun": False,
            }
        if most_similar_word:
            return {
                "is_Hetergraphic_Pun": True,
                "pun_word": most_similar_word['original word'],
                "similar_sounding_word": most_similar_word['similar sounding'],
                "definition": find_most_similar_definition(
                    [sentence, most_similar_word['original word']], most_similar_word
                ),
            }
        else:
            return {
                "is_Hetergraphic_Pun": False,
            }

    else:
        return {
            "is_Hetergraphic_Pun": False,
        }


if __name__ == '__main__':
    print(detect_pun("Why is it so wet in England? Because many kings and queens have reigned there."))
