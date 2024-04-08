import pandas as pd
from gensim.models import Word2Vec, FastText
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('path_to_dataset.csv')

# Train Word2Vec and FastText embeddings on the sentences
sentences = df['sentence'].apply(lambda x: x.split()).tolist()
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Extract word embeddings for the word or phrase
df['w2v'] = df['word_or_phrase'].apply(lambda x: w2v_model.wv[x] if x in w2v_model.wv else [0]*100)
df['ft'] = df['word_or_phrase'].apply(lambda x: ft_model.wv[x] if x in ft_model.wv else [0]*100)

# Extract POS tagging for the word or phrase
df['pos'] = df['word_or_phrase'].apply(lambda x: pos_tag([x])[0][1])

# Encode POS tags
pos_dummies = pd.get_dummies(df['pos'], prefix='pos')
df = pd.concat([df, pos_dummies], axis=1)

# Assuming `sense` is categorical, encode it as target variable
df['target'] = df['sense'].astype('category').cat.codes

# Define X and y for the model
X = pd.concat([df['w2v'].apply(pd.Series), df['ft'].apply(pd.Series), pos_dummies], axis=1)
y = df['target']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
