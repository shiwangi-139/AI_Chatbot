Data Science & Analytics Internship Dynamic AI Chatbot – Project 2

📁 Project Files 
File                                  Purpose 
intents.json                         Training data 
PersonalizedChatBot.ipynb            Main Jupyter Notebook 
app.py                               Streamlit web app  
words.pkl                            Generated after training 
classes.pkl                          Generated after training 
chatbot_model.h5                     Generated after training

🚀 How to Run 

Local Machine bash# Install dependencies pip install tensorflow streamlit nltk

#First run the notebook to train the model jupyter notebook PersonalizedChatBot.ipynb

# run the streamlit app streamlit run app.py

🧠 How It Works 
         User Input 
             ↓ 
        Tokenization (NLTK) 
             ↓ 
        Lemmatization 
             ↓ 
        Bag of Words 
             ↓ 
        Neural Network (TensorFlow) 
             ↓ 
        Intent Classification 
             ↓ 
        Random Response Selection 
             ↓ 
        Bot Reply

📊 Technology Stack

Python — basic language 
NLTK — tokenization and lemmatization
TensorFlow/Keras - model neuralnet
Streamlit — web ui 
JSON — intent data storage
Pickle — persistence of model


💬 Chatbot Knowledge Topics

Data Science 
Machine Learning 
Deep Learning & Neural Networks
NLP Cryptocurrency & Blockchain 
Time Series Analysis (ARIMA, LSTM, Prophet) 
Python libraries 
Internship info
