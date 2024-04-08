from HomographicPun import main as homographic_pun_main
from HetergraphicPun import detect_pun as heterographic_pun_main
sentence_List = [
    "I'm friends with all the electricians, because we have good current connections.",
    'The crane picked up the heavy load.',
    'The boating store had its best sail ever.',
    "Beware of the false prophets , who come to you in sheep's clothing , and inwardly are ravening wolves .",
    "Don't put the cart before the horse .",
    'Alex , I\'ll take " Things Only I Know " for $ 1000 , please .',
    'Good riding at two anchors , men have told , for if the one fails , the other may hold .',
    'While the grass grows the steed starves .',
    'Our greatest glory is not in never falling but in rising every time we fall .',
    'I am the Marlboro Man of Borg . I ( cough ) will assim - ( choke , wheeze ) . .',
    'There is nothing more precious than time and nothing more prodigally wasted .',
    'He complains wrongfully at the sea that suffer shipwreck twice .',
    "Life's battle don't always go to the stronger or faster man , but sooner or later the man who wins is the one who thinks he can .",
    'An egotist is a person who is more interested in himself than in me .',
    'Press any key to continue . . Any other key to format hard drive .', 'You can catch more flies with a drop of honey than with a barrel of vinegar .', 'They who would be young when they are old must be old when they are young .', 'If fortune favours , beware of being exalted ; if fortune thunders , beware of being overwhelmed .', 'A friend cannot be known in prosperity nor an enemy be hidden in adversity .', 'If it looks like a duck , swims like a duck , and quacks like a duck , then it probably is a duck .', 'Give a man a fish and you feed him for a day . Teach a man to fish and you feed him for a lifetime .', 'For want of a nail the shoe is lost , for want of a shoe the horse is lost , for want of a horse the rider is lost .', "Love is not finding someone to live with ; it's finding someone whom you can't live without .", 'Autobiography: when your car starts telling you about its life.?', 'OLD GEOGRAPHERS never die , they just become legends.', "Why so glum?  I've got a bad case of shingles so the doctor prescribed aluminum siding. ", 'Time flies like an arrow; fruit flies like a banana.', 'You can tune a guitar, but you can’t tuna fish. Unless of course, you play bass.', 'Those who like sport fishing can really get hooked .', 'A bank manager who was also a high jumper spent most of his time in the vault .', 'Getting rid of your boat for another could cause a whole raft of problems .', "Lawyers have to like alcohol because they're always being called to the bar .", 'The man brought an umbrella with him into the ice cream store because he heard there was a chance of sprinkles .', 'In the air duct installers union they have lots of opportunity to vent .', 'For a while , Houdini used a lot of trap doors in his act , but he was just going through a stage .', 'Her exam was on the human skeleton , so she decided to bone up .', 'Junior loved being a member of the wrestling team even though he was prone to lose .', 'I travel all over America , Tom stated .', 'Why is it so wet in England? Because many kings and queens have reigned there.', 'You can tune a guitar, but you can’t tuna fish. Unless of course, you play bass.', 'If you burn the candle on both ends , you’re not as bright as you think', 'The boating store had its best sail ever.', 'People who like gold paint have a gilt complex.', "A bicycle can't stand on its own because it is two-tired.", 'No matter how much you push the envelope, it will still be stationery.', "A pessimist's blood type is always B negative.", 'Two peanuts walk into a bar, and one was a salted.', 'Reading while sunbathing makes you well red.', 'When a clock is hungry it goes back four seconds.', 'When she saw her first strands of gray hair, she thought she’d dye.', 'When fish are in schools they sometimes take debate.', 'With her marriage she got a new name and a dress.', "The pony couldn't speak - he was a little hoarse."]
answerDict = {}
for sentence in sentence_List:
    print(f"Sentence to detect pun: {sentence}")
    homographic_result = homographic_pun_main(sentence)
    if homographic_result['is_Homographic_Pun']:
        answerDict[sentence] = {
            'is_Pun': True,
            'Pun Type': 'Homographic Pun',
            'pun_word': homographic_result['pun_word'],
            'first_definition': homographic_result['first_definition'],
            'second_definition': homographic_result['second_definition']
        }
        continue
    heterographic_result = heterographic_pun_main(sentence)
    if heterographic_result['is_Hetergraphic_Pun']:
        answerDict[sentence] = {
            'is_Pun': True,
            'Pun Type': 'Heterographic Pun',
            'pun_word': heterographic_result['pun_word'],
            'similar_sounding_word': heterographic_result['similar_sounding_word'],
            'first_definition': heterographic_result['definition']['original word']['definition'],
            'second_definition': heterographic_result['definition']['similar sounding']['definition']
        }
        continue
    answerDict[sentence] = {
        'is_Pun': False,
    }
print(answerDict)