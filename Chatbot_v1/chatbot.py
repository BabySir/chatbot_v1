#Implement the chatbot logic that takes user input, preprocesses it, 
# predicts the intent, and returns an appropriate response.


#python preprocess.py
#python train_model.py
#python chatbot.py
import json
import numpy as np
import random
import tensorflow as tf
from keras import layers, models 
import spacy
import string
import nltk

from preprocess import words, classes, label_encoder, stop_words

# Initialize SpaCy
nlp = spacy.load('en_core_web_sm')

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Load intents
with open('intents.json') as file:
    data = json.load(file)

def preprocess_input(sentence):
    # Tokenize and lemmatize
    doc = nlp(sentence.lower())
    tokens = [token.lemma_ for token in doc if token.text not in string.punctuation and not token.is_stop]
    # Create bag of words
    bag = [1 if word in tokens else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
    bow = preprocess_input(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    if not return_list:
        return_list.append({"intent": "noanswer", "probability": "1.0"})
    return return_list

def get_response(intents_list):
    tag = intents_list[0]['intent']
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I don't understand."

def chatbot_response(text):
    intents = predict_class(text)
    response = get_response(intents)
    return response



# Simple command-line interface
if __name__ == "__main__":
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        response = chatbot_response(inp)
        print(f"Bot: {response}")