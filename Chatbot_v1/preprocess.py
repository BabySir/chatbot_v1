import json
import nltk
import numpy as np
import random
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import spacy
import string

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize SpaCy
nlp = spacy.load('en_core_web_sm')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = set(string.punctuation)

# Preprocess the data
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern
        doc = nlp(pattern.lower())
        tokens = [token.lemma_ for token in doc if token.text not in ignore_letters and not token.is_stop]
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Remove duplicates and sort
words = sorted(set(words))
classes = sorted(set(classes))

# Create training data
X = []
y = []

for (pattern_tokens, tag) in documents:
    # Create a bag of words for each pattern
    bag = [1 if word in pattern_tokens else 0 for word in words]
    X.append(bag)
    y.append(tag)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(classes)

# Convert to numpy arrays
X = np.array(X)
y = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)