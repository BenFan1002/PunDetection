from nltk.corpus import wordnet as wn


def get_hyponyms(synset, depth=1):
    """Helper function to fetch hyponyms up to a certain depth."""
    if depth <= 0:
        return []

    hyponyms = synset.hyponyms()
    for hypo in hyponyms[:]:  # Copy the list to prevent in-place modifications
        hyponyms.extend(get_hyponyms(hypo, depth - 1))

    return hyponyms


def context_relevant_definitions(word, context_words):
    word_synsets = wn.synsets(word)
    if not word_synsets:
        return None

    definitions = {}

    for context_word in context_words:
        best_definition = None
        for w_syn in word_synsets:
            if best_definition:  # Break if already found
                break

            # Check definition and example of word
            if context_word in w_syn.definition():
                best_definition = w_syn.definition()
                break
            for ex in w_syn.examples():
                if context_word in ex:
                    best_definition = w_syn.definition()
                    break

            # Check up to 5 levels of hyponyms
            for level in range(1, 6):
                for hypo in get_hyponyms(w_syn, level):
                    if best_definition:  # Break if already found
                        break
                    if context_word in hypo.definition():
                        best_definition = hypo.definition()
                        break
                    for ex in hypo.examples():
                        if context_word in ex:
                            best_definition = hypo.definition()
                            break

        definitions[context_word] = best_definition if best_definition else "No relevant definition found"

    return definitions


# Sample call
word = "ticks"
context_words = ["watch", "bug"]
definitions = context_relevant_definitions(word, context_words)
for context_word, definition in definitions.items():
    print(f"Most relevant definition for '{word}' in the context of '{context_word}': {definition}")
