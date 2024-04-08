import re


def parse_hierarchy(hierarchy):
    lines = hierarchy.split("\n")
    tree = {}
    current_path = []
    for line in lines:
        # Count the number of tabs to determine the depth
        depth = line.count('\t')
        current_path = current_path[:depth]
        node = line.strip().capitalize()
        node = re.match(r'^(\w+)', node).group(1)
        current_node = tree
        for ancestor in current_path:
            current_node = current_node[ancestor]
        current_node[node] = {}
        current_path.append(node)
    return tree


def find_common_ancestor_and_merge(tree1, tree2):
    if not isinstance(tree1, dict) or not isinstance(tree2, dict):
        return tree1, tree2

    merged_tree = {}
    for key in set(tree1.keys()) | set(tree2.keys()):
        t1_node, t2_node = tree1.get(key), tree2.get(key)

        if t1_node is None:
            merged_tree[key] = t2_node
        elif t2_node is None:
            merged_tree[key] = t1_node
        else:
            merged_tree[key] = find_common_ancestor_and_merge(t1_node, t2_node)

    return merged_tree


def print_hierarchy(tree, indent=0):
    for key in tree:
        print('\t' * indent + key)
        print_hierarchy(tree[key], indent + 1)


def find_path(tree, word, path=[]):
    """Find path from root to the word in the tree."""
    if word in tree:
        return path + [word]
    for key, subtree in tree.items():
        result = find_path(subtree, word, path + [key])
        if result:
            return result
    return None


def distance_and_path(tree, word1, word2):
    """Calculate the distance and path from word1 to word2 in the tree."""
    path1 = find_path(tree, word1)
    path2 = find_path(tree, word2)
    if path1 is None or path2 is None:
        return None, None  # One of the words is not in the tree

    # Find the length of the common prefix of path1 and path2
    common_length = 0
    for p1, p2 in zip(path1, path2):
        if p1 == p2:
            common_length += 1
        else:
            break

    # Calculate distance
    distance = (len(path1) - common_length) + (len(path2) - common_length)

    # Determine the path from word1 to word2
    path = path1[common_length:][::-1] + path2[common_length - 1:]

    return distance, path


animal1, animal2, animal3, animal4 = 'crane', 'flamingo', 'seal', 'horse'
text = "(animal1, animal2), (animal1, animal3), (animal1, animal4), (animal2, animal3), (animal2, animal4), (animal3, animal4)."
hierarchies = {
    'crane': """organism.n.01
	 animal.n.01
		 chordate.n.01
			 vertebrate.n.01
				 bird.n.01
					 aquatic_bird.n.01
						 wading_bird.n.01
							 crane.n.05""",
    'flamingo': """organism.n.01
	 animal.n.01
		 chordate.n.01
			 vertebrate.n.01
				 bird.n.01
					 aquatic_bird.n.01
						 wading_bird.n.01
							 flamingo.n.01""",
    'seal': """organism.n.01
	 animal.n.01
		 chordate.n.01
			 vertebrate.n.01
				 mammal.n.01
					 placental.n.01
						 aquatic_mammal.n.01
							 pinniped_mammal.n.01
								 seal.n.09""",
    'horse': """organism.n.01
	 animal.n.01
		 chordate.n.01
			 vertebrate.n.01
				 mammal.n.01
					 placental.n.01
						 ungulate.n.01
							 odd-toed_ungulate.n.01
								 equine.n.01
									 horse.n.01""",

}
# Remove the final period and split the text by "),"
pairs = text.rstrip('.').split('), ')
def bump_up(d):
    while len(next(iter(d.values()))) == 1:
        d = next(iter(d.values()))
    return d
# Add back the closing parenthesis to each pair, except for the last one
pairs = [pair + ')' for pair in pairs]
for pair in pairs:
    pair = pair.split(", ")
    first_animal = eval(pair[0].replace("(", ""))
    second_animal = eval(pair[1].replace(")", ""))
    print("similarity for the pair ({}, {}) using Wordnet:".format(
        re.match(r'^(\w+)', first_animal).group(1),
        re.match(r'^(\w+)', second_animal).group(1)
    ))
    merged_hierarchy = find_common_ancestor_and_merge(parse_hierarchy(hierarchies[first_animal]), parse_hierarchy(hierarchies[second_animal]))
    print_hierarchy(bump_up(merged_hierarchy))
    distance, path = distance_and_path(merged_hierarchy, first_animal.capitalize(), second_animal.capitalize())
    print(f"Path: ", "->".join(path))
    print(f"Distance: {distance}")
    print("Path Similarity: ", round(1 / (distance + 1), 2))
    print()