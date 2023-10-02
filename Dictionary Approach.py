import pyperclip
from nltk.corpus import wordnet
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

skeleton = {
    "the arrangement of parts that gives something its basic form": ['structure', 'framework', 'architecture', 'shell',
                                                                     'frame', 'infrastructure', 'fabric',
                                                                     'configuration', 'armature', 'shape', 'edifice',
                                                                     'framing', 'network', 'outline', 'cage', 'cadre',
                                                                     'lattice', 'contour', 'profile', 'silhouette',
                                                                     'figure', 'chassis']

}
guts = {
    "the internal organs of the body": ['inside(s)', 'entrails', 'viscera', 'innards', 'vitals', 'inwards',
                                        'intestine(s)', 'bowel(s)', 'chitlins', 'chitterlings', 'variety meat',
                                        'giblet(s)'],
    "strength of mind to carry on in spite of danger": ['courage', 'bravery', 'heroism', 'courageousness',
                                                        'gallantry', 'prowess', 'nerve', 'daring', 'heart', 'valor',
                                                        'fearlessness', 'hardihood', 'pecker', 'stoutness',
                                                        'gutsiness', 'virtue', 'intrepidity', 'gumption', 'bottle',
                                                        'moxie', 'grit', 'fortitude', 'daringness', 'intrepidness',
                                                        'dauntlessness', 'spunk', 'determination', 'backbone',
                                                        'stamina', 'intestinal fortitude', 'mettle',
                                                        'greatheartedness', 'pluck', 'perseverance', 'doughtiness',
                                                        'fiber', 'tenacity', 'stomach', 'temper', 'endurance',
                                                        'resolution', 'audacity', 'gall', 'temerity', 'boldness',
                                                        'cojones', 'pluckiness', 'effrontery', 'cheek',
                                                        'brazenness'],
    "the strength of mind that enables a person to endure pain or hardship": ['courage', 'grit', 'fortitude',
                                                                              'bravery', 'spunk', 'stamina',
                                                                              'backbone', 'courageousness', 'pluck',
                                                                              'determination', 'fiber',
                                                                              'grittiness', 'endurance', 'nerve',
                                                                              'intestinal fortitude', 'constancy',
                                                                              'daring', 'resoluteness',
                                                                              'resolution', 'valor', 'fearlessness',
                                                                              'tolerance', 'gallantry',
                                                                              'purposefulness', 'audacity',
                                                                              'chutzpah', 'stoutness', 'mettle',
                                                                              'gall', 'intrepidity', 'temerity',
                                                                              'boldness', 'brass', 'heart',
                                                                              'sufferance', 'hutzpah',
                                                                              'intrepidness', 'effrontery',
                                                                              'dauntlessness', 'spirit', 'chutzpa',
                                                                              'forbearance', 'hardihood',
                                                                              'greatheartedness', 'cheek',
                                                                              'doughtiness', 'hutzpa', 'nerviness'],
    "to take the internal organs out of": ['cleans', 'bones', 'draws', 'removes', 'extracts', 'disembowels',
                                           'eviscerates', 'cuts', 'excises', 'dresses', 'transplants', 'withdraws',
                                           'yanks']
}
fight = {
    "to oppose (someone) in physical conflict": ['battle', 'combat', 'war (against)', 'duel', 'beat', 'clash (with)',
                                                 'wrestle', 'skirmish (with)', 'hit', 'punch', 'strike', 'knock',
                                                 'slap', 'joust', 'brawl', 'scrimmage (with)', 'smack', 'box', 'pound',
                                                 'smite', 'slug', 'whack', 'belt', 'spar', 'bang', 'clobber',
                                                 'bludgeon', 'bat', 'hammer', 'slam', 'swipe', 'swat', 'tussle',
                                                 'batter', 'thump', 'buffet', 'bash', 'bop', 'sock', 'paste', 'slog',
                                                 'grapple', 'bump', 'whale', 'scuffle', 'wallop', 'collide', 'thwack'],
    "to strive to reduce or eliminate": ['oppose', 'combat', 'battle', 'counter', 'resist', 'confront',
                                         'contend (with)', 'withstand', 'thwart', 'oppugn', 'frustrate', 'face', 'foil',
                                         'baffle', 'checkmate', 'defy', 'meet'],
    "to engage in a contest": ['compete', 'contend', 'race', 'battle', 'rival', 'vie', 'play', 'challenge', 'face off',
                               'engage', 'maneuver', 'try out', 'work', 'go out', 'jostle', 'train', 'jockey'],
    "to express different opinions about something often angrily": ['bicker', 'argue', 'quarrel', 'spat', 'brawl',
                                                                    'clash', 'dispute', 'squabble', 'debate', 'wrangle',
                                                                    'fall out', 'scrap', 'row', 'controvert', 'discuss',
                                                                    'quibble', 'jar', 'hassle', 'mix it up',
                                                                    'lock horns', 'argufy', 'altercate', 'brabble',
                                                                    'contest', 'butt heads', 'tiff', 'challenge',
                                                                    'contend', 'dare', 'bandy words', 'protest', 'defy',
                                                                    'tangle', 'kick', 'object', 'fuss', 'cavil',
                                                                    'nitpick', 'consider'],
    "to refuse to give in to": ['resist', 'oppose', 'withstand', 'repel', 'combat', 'defy', 'contest', 'challenge',
                                'buck', 'thwart', 'contend (with)', 'counter', 'battle', 'dispute', 'contradict',
                                'hinder', 'stem', 'check', 'frustrate', 'balk', 'obstruct', 'foil', 'baffle'],
    "a physical dispute between opposing individuals or groups": ['battle', 'skirmish', 'clash', 'struggle', 'tussle',
                                                                  'scuffle', 'brawl', 'fray', 'combat', 'contest',
                                                                  'confrontation', 'conflict', 'duel', 'dustup',
                                                                  'altercation', 'scrimmage', 'fracas', 'scrap',
                                                                  'quarrel', 'scrum', 'blows', 'pitched battle',
                                                                  'dispute', 'mêlée', 'melee', 'spat', 'hassle',
                                                                  'battle royal', 'fistfight', 'rough-and-tumble',
                                                                  'fisticuffs', 'slugfest', 'squabble', 'tangle',
                                                                  'joust', 'broil', 'punch-up', 'affray', 'row',
                                                                  'ruckus', 'donnybrook', 'mix-up', 'face-off',
                                                                  'wrangle', 'grapple', 'controversy', 'free-for-all',
                                                                  'punch-out', 'falling-out', 'catfight', 'ruction',
                                                                  'cross fire', 'argument', 'misunderstanding', 'tiff',
                                                                  'disagreement', 'contretemps', 'kickup', 'handgrips',
                                                                  'argle-bargle', 'argy-bargy'],
    "a forceful effort to reach a goal or objective": ['battle', 'struggle', 'effort', 'fray', 'throes', 'attempt',
                                                       'scrabble', 'grind', 'try', 'work', 'pains', 'war', 'combat',
                                                       'endeavor', 'exertion', 'toil', 'trouble', 'warfare', 'labor',
                                                       'tussle', 'conflict', 'strife', 'drudgery', 'contest', 'sweat',
                                                       'travail', 'essay'],
    "an inclination to fight or quarrel": ['aggression', 'aggressiveness', 'hostility', 'assaultiveness', 'defiance',
                                           'combativeness', 'belligerence', 'pugnacity', 'belligerency', 'militancy',
                                           'militance', 'disputatiousness', 'contentiousness', 'quarrelsomeness',
                                           'feistiness', 'scrappiness', 'bellicosity', 'truculence', 'militantness',
                                           "chip on one's shoulder", 'antagonism', 'fierceness', 'irritableness',
                                           'imperialism', 'irritability', 'unfriendliness', 'militarism',
                                           'hyperaggressiveness', 'jingoism', 'captiousness', 'testiness',
                                           'fractiousness', 'disagreeableness', 'crossness', 'rudeness', 'peevishness',
                                           'crankiness', 'waspishness', 'grumpiness', 'petulance', 'biliousness',
                                           'pettishness', 'surliness', 'fretfulness', 'grouchiness', 'querulousness',
                                           'acidity', 'irascibleness', 'orneriness', 'irascibility', 'huffiness'],
    "an often noisy or angry expression of differing opinions": ['dispute', 'quarrel', 'altercation', 'bicker',
                                                                 'controversy', 'disagreement', 'argument', 'brawl',
                                                                 'misunderstanding', 'row', 'skirmish', 'squabble',
                                                                 'spat', 'debate', 'wrangle', 'battle royal', 'feud',
                                                                 'clash', 'disputation', 'set-to', 'falling-out',
                                                                 'imbroglio', 'cross fire', 'tiff', 'donnybrook',
                                                                 'contretemps', 'kickup', 'rhubarb', 'tussle', 'scrap',
                                                                 'difference', 'tangle', 'attack', 'run-in',
                                                                 'argle-bargle', 'protest', 'dissention', 'hassle',
                                                                 'argy-bargy', 'contention', 'vendetta', 'objection',
                                                                 'fisticuffs', 'melee', 'mêlée', 'protestation',
                                                                 'logomachy', 'dissension', 'catfight', 'fuss',
                                                                 'affray', 'fray', 'fracas', 'free-for-all']
}


def find_similarity(wordDict1, wordDict2):
    max_avg_similarity = 0
    max_avg_similarity_def_pair = ()
    # Iterate over the dictionary
    for def1, synonyms1 in wordDict1.items():
        for def2, synonyms2 in wordDict2.items():
            total_similarity = 0
            count = 0
            for syn1 in synonyms1:
                for syn2 in synonyms2:
                    if syn1 in model and syn2 in model:  # Checking if the words are in the model vocabulary
                        similarity = cosine_similarity([model[syn1]], [model[syn2]])[0][0]
                        total_similarity += similarity
                        count += 1
            if count > 0:
                avg_similarity = total_similarity / count
                if avg_similarity > max_avg_similarity:
                    max_avg_similarity = avg_similarity
                    max_avg_similarity_def_pair = (def1, def2)

    print(
        f"The definition with the highest average similarity to 'gut', 'skeleton' is: '{max_avg_similarity_def_pair}' with "
        f"an average similarity of {max_avg_similarity}")
    return max_avg_similarity_def_pair, max_avg_similarity


print(find_similarity(guts, skeleton))
print(find_similarity(guts, fight))
print(find_similarity(skeleton, fight))
