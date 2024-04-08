import fuzzy

soundex = fuzzy.Soundex(4)
word = "example"
phonetic_representation = soundex(word)
