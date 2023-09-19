import string

import torch
from nltk import word_tokenize, MWETokenizer
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForMaskedLM


def mask_and_predict(sentence, top_n=5):
    # Load pre-trained model and tokenizer
    model_name = 'bert-base-uncased'
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()

    words = sentence.split()
    results = []

    # Loop through each word in the sentence
    for i in range(len(words)):
        masked_sentence = ' '.join([words[j] if j != i else '[MASK]' for j in range(len(words))])
        inputs = tokenizer(masked_sentence, return_tensors="pt")

        # Get model output
        with torch.no_grad():
            output = model(**inputs)
            logits = output.logits

        # Get top_n predictions for the masked word
        masked_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()
        probs = logits[0, masked_index].softmax(dim=0)
        top_prob_values, top_indices = torch.topk(probs, top_n)

        print(f"Original Word: {words[i]}")
        for idx, token_id in enumerate(top_indices):
            predicted_token = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
            prob_value = top_prob_values[idx].item()
            print(f"   Candidate {idx + 1}: {predicted_token} with probability {prob_value:.4f}")

        top_prediction = tokenizer.convert_ids_to_tokens([top_indices[0].item()])[0]
        results.append((words[i], top_prediction, top_prob_values[0].item()))

    return results


def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))


sentence = """If you burn the candle on both ends, you’re not as bright as you think."""
tokenizer = MWETokenizer()  # Multi-word expression tokenizer
tokens = tokenizer.tokenize(word_tokenize(sentence))
stop_words = set(stopwords.words('english'))
# filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
# print(" ".join(filtered_tokens))
predictions = mask_and_predict(" ".join(tokens))
