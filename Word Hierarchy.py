from nltk.corpus import wordnet as wn


class Node:
    def __init__(self, synset):
        self.synset = synset
        self.children = []

    def add_children(self, children):
        self.children.extend(children)


def build_hypernym_trees(synsets, depth=0, max_depth=15):
    """Recursively build a forest of hypernym trees."""
    nodes = []
    if depth > max_depth or "organism.n.01" in [x.name() for x in synsets[0].hyponyms()]:
        return None
    for synset in synsets:
        node = Node(synset)
        for hypernym in synset.hypernyms():
            children = build_hypernym_trees([hypernym], depth + 1, max_depth)
            if children is not None:
                node.add_children(children)
        nodes.append(node)
    return nodes


def traverse_trees(nodes, tree_path=[], path=[]):
    """Traverse the trees interactively."""
    if not tree_path:
        # If traversing from the roots, allow user to select tree
        print(f"Select a tree to traverse:")
        for i, node in enumerate(nodes):
            print(f"{i + 1}. {node.synset.name()} - {node.synset.definition()}")
        choice = int(input("Choose a tree (enter the number): "))
        traverse_trees(nodes[choice - 1], tree_path=[nodes[choice - 1].synset.name()], path=[])

    else:
        node = nodes
        path.append(node.synset.name())  # Add the current node to the path
        if not node.children:
            print("Reached the top of this branch.")
            print("Traversal Path:")
            for i, synset_name in enumerate(reversed(path)):
                print("\t" * (i), f"{synset_name}")
            return
        print(f"Hypernyms of {node.synset.name()}:")
        if len(node.children) == 1:
            child = node.children[0]
            print(f"1. {child.synset.name()} - {child.synset.definition()}")
            print("Automatically proceeding as there is only one branch.")
            traverse_trees(child, tree_path=tree_path, path=path)
        elif node.children:
            for i, child in enumerate(node.children):
                print(f"{i + 1}. {child.synset.name()} - {child.synset.definition()}")
            choice = int(input("Choose a branch (enter the number): "))
            traverse_trees(node.children[choice - 1], tree_path=tree_path, path=path)


# Assume we're working with the first synset of 'dog'
synset = wn.synsets('horse')

# Build the tree and traverse interactively
root = build_hypernym_trees(synset, max_depth=10)
traverse_trees(root)
