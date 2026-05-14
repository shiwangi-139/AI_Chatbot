import streamlit as st
import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt_tab", quiet=True)

lemmatizer = WordNetLemmatizer()

# ── Load intents & train model on startup ─────────────────────
@st.cache_resource
def load_chatbot():
    with open("intents.json") as f:
        intents = json.load(f)

    words     = []
    classes   = []
    documents = []
    ignore_chars = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(set([
        lemmatizer.lemmatize(w.lower())
        for w in words if w not in ignore_chars
    ]))
    classes = sorted(set(classes))

    # Build training data (bag of words)
    training = []
    for doc in documents:
        word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
        bag = [1 if w in word_patterns else 0 for w in words]
        output = [0] * len(classes)
        output[classes.index(doc[1])] = 1
        training.append([bag, output])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    X = np.array(list(training[:, 0]))
    y = np.argmax(np.array(list(training[:, 1])), axis=1)

    # Train Random Forest classifier (no tensorflow needed!)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)

    return clf, words, classes, intents

clf, words, classes, intents = load_chatbot()

# ── Prediction functions ───────────────────────────────────────
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return np.array([1 if w in sentence_words else 0 for w in words])

def chatbot_response(text):
    bow      = bag_of_words(text).reshape(1, -1)
    proba    = clf.predict_proba(bow)[0]
    max_prob = np.max(proba)

    if max_prob < 0.25:
        return "I'm not sure I understand. Could you rephrase that?"

    predicted_class = classes[np.argmax(proba)]
    for intent in intents["intents"]:
        if intent["tag"] == predicted_class:
            return random.choice(intent["responses"])

    return "I'm not sure I understand. Could you rephrase that?"

# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Dynamic AI Chatbot")
st.caption("Powered by Python · NLTK · Machine Learning ")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI assistant. Ask me about data science, machine learning, cryptocurrency, or just say hi! 😊"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    response = chatbot_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

with st.sidebar:
    st.header("About this Chatbot")
    st.info("""
    **Tech Stack:**
    - Python
    - NLTK (NLP)
    - Scikit-learn (ML)
    - Streamlit (UI)

    **Topics I know:**
    - Data Science & ML
    - Deep Learning / NLP
    - Cryptocurrency
    - Time Series
    - Python Libraries
    """)
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! How can I help you?"}
        ]
        st.rerun()
