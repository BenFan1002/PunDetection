from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download

# Download required resources
download('punkt')
download('stopwords')

# Set of English stopwords
stop_words = set(stopwords.words('english'))


def token_overlap(text1, text2):
    """Count the overlap in terms of tokens between two texts."""
    tokens1 = set([word for word in word_tokenize(text1.lower()) if word.isalnum() and word not in stop_words])
    tokens2 = set([word for word in word_tokenize(text2.lower()) if word.isalnum() and word not in stop_words])
    return len(tokens1.intersection(tokens2))


def context_relevant_definitions(word, context_words):
    word_synsets = wn.synsets(word)
    if not word_synsets:
        return None

    definitions = {}

    for context_word in context_words:
        best_definition = None
        max_overlap = -1
        for w_syn in word_synsets:
            overlap = token_overlap(w_syn.definition(), context_word)
            if overlap > max_overlap:
                max_overlap = overlap
                best_definition = w_syn.definition()

            # Checking hyponyms up to 5 levels
            for hypo in get_hyponyms(w_syn, 5):
                overlap = token_overlap(hypo.definition(), context_word)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_definition = hypo.definition()

        definitions[context_word] = best_definition if best_definition else "No relevant definition found"

    return definitions


# Helper function to fetch hyponyms up to a certain depth
def get_hyponyms(synset, depth=1):
    if depth <= 0:
        return []

    hyponyms = synset.hyponyms()
    for hypo in hyponyms[:]:  # Copy the list to prevent in-place modifications
        hyponyms.extend(get_hyponyms(hypo, depth - 1))

    return hyponyms


# Sample call
word = "ticks"
context_words = ["watch", "bug"]
definitions = context_relevant_definitions(word, context_words)
for context_word, definition in definitions.items():
    print(f"Most relevant definition for '{word}' in the context of '{context_word}': {definition}")
