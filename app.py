import streamlit as st
import json
import pickle
import random
import numpy as np
import nltk
import os
from nltk.stem import WordNetLemmatizer
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model

# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Load model and data ────────────────────────────────────────
@st.cache_resource
def load_chatbot():
    lemmatizer = WordNetLemmatizer()
    model      = load_model("chatbot_model.h5")
    words      = pickle.load(open("words.pkl",   "rb"))
    classes    = pickle.load(open("classes.pkl", "rb"))
    with open("intents.json") as f:
        intents = json.load(f)
    return lemmatizer, model, words, classes, intents

lemmatizer, model, words, classes, intents = load_chatbot()

# ── Chatbot functions ──────────────────────────────────────────
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def chatbot_response(text):
    bow = bag_of_words(text)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = sorted(
        [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD],
        key=lambda x: x[1], reverse=True
    )
    if not results:
        return "I'm not sure I understand. Could you rephrase that?"
    tag = classes[results[0][0]]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure I understand. Could you rephrase that?"


st.title("🤖 Dynamic AI Chatbot")
st.caption("Powered by Python · NLTK · Deep Learning | Amdox Internship Project")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI assistant. Ask me about data science, machine learning, cryptocurrency, or just say hi! 😊"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response
    response = chatbot_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Sidebar
with st.sidebar:
    st.header("About this Chatbot")
    st.info("""
    **Tech Stack:**
    - Python
    - NLTK (NLP)
    - TensorFlow/Keras
    - Streamlit (UI)

    **Topics I know:**
    - Data Science
    - Machine Learning
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