import gensim.downloader as api

# Load pretrained model (note: this will consume a lot of memory!)
model = api.load("word2vec-google-news-300")
animal1, animal2, animal3, animal4 = 'crane', 'flamingo', 'seal', 'horse'
text = "(animal1, animal2), (animal1, animal3), (animal1, animal4), (animal2, animal3), (animal2, animal4), (animal3, animal4)."
# Remove the final period and split the text by "),"
pairs = text.rstrip('.').split('), ')

# Add back the closing parenthesis to each pair, except for the last one
pairs = [pair + ')' for pair in pairs]
for pair in pairs:
    pair = pair.split(", ")
    first_animal = eval(pair[0].replace("(", ""))
    second_animal = eval(pair[1].replace(")", ""))
    print(f"similarity for the pair ({first_animal}, {second_animal}) using Word2Vec:")
    print(model.similarity(first_animal, second_animal))
