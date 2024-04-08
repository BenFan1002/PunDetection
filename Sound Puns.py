import numpy as np

from weighted_levenshtein import lev, osa, dam_lev


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
        transcription = cmu_dict[input_word]
    print(transcription)
    # If the word is not in the dictionary, return a message
    if not transcription:
        return f"The word '{input_word}' was not found in the CMU Pronouncing Dictionary."

    # Calculate the edit distance for each transcription of the input word
    # against all other words in the dictionary
    similarity_scores = []
    for word, word_transcription in cmu_dict.items():
        # word_transcription = word_transcription.split(" ")
        # if len(word_transcription) != len(transcription):
        #     continue
        if transcription != word_transcription:
            score = lev(transcription, word_transcription,
                        insert_costs=np.full(128, 2.0, dtype=np.float64),
                        delete_costs=np.full(128, 2.0, dtype=np.float64),
                        )
            if word == "ANTELOPE":
                print("transcription: ", f"{transcription}", "word_transcription: ", f"{word} {word_transcription}",
                      "score: ", score)
            similarity_scores.append((score, word))

    # Sort the words by the edit distance and get the top results
    similarity_scores.sort()
    similar_words = [word for score, word in similarity_scores[:num_results]]

    return similar_words


# Attempting to read the CMUDict file using 'ISO-8859-1' encoding
cmu_dict = {}

# Trying to open the file with a different encoding if UTF-8 fails
try:
    with open("cmudict-0.7b", 'r', encoding='ISO-8859-1') as file:
        for line in file:
            if not line.startswith(";;;"):  # Skip comment lines
                parts = line.strip().split('  ')
                if len(parts) == 2:
                    word, transcription = parts
                    cmu_dict[word] = transcription
except UnicodeDecodeError as e:
    error_message = str(e)
sample_word = "Cantaloupe"
# Set all insert costs to 2
insert_costs = np.full(128, 2.0, dtype=np.float64)  # now all insertions have a cost of 2
prouncation_set = set()
for key, item in cmu_dict.items():
    for i in item.split(" "):
        prouncation_set.add(i)
print(sorted(list(prouncation_set)))
# Now you can use insert_costs in the weighted Levenshtein function calls
# For example:
# print(lev(cmu_dict[sample_word.upper()], cmu_dict['sample'.upper()], insert_costs=insert_costs))
# print(cmu_dict[sample_word.upper()] + '\n' + cmu_dict['sample'.upper()])
# similar_words = find_similar_sounding_words(sample_word, cmu_dict, num_results=5)
# print(f"Words similar to '{sample_word}':")
# for word in similar_words:
#     print(f"    {word}")
