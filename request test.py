import requests

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "i have a good day today",
})
print(output[0][0]['label'])