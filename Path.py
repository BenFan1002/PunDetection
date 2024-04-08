from nltk.corpus import wordnet as wn


def find_path(word1, word2):
    # Get the synsets for each word
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    for syn1 in synsets1:
        for syn2 in synsets2:
            # Find the shortest path between the synsets
            path = syn1.shortest_path_distance(syn2)
            if path is not None:
                # Get the hypernym paths for each synset
                paths1 = syn1.hypernym_paths()
                paths2 = syn2.hypernym_paths()

                # Find the common path
                for p1 in paths1:
                    for p2 in paths2:
                        common = set(p1).intersection(p2)
                        if common:
                            # Print the path from word1 to word2
                            path1 = p1[:p1.index(list(common)[0]) + 1]
                            path2 = p2[:p2.index(list(common)[0]) + 1][::-1]
                            full_path = path1 + path2[1:]
                            return [synset.name().split('.')[0] for synset in full_path]


# Example usage
path = find_path('dog', 'cat')
print(path)