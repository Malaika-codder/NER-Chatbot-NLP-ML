import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="NER Chatbot", layout="centered")

# -----------------------------
# Custom CSS (Purple + Black + Pink)
# -----------------------------
st.markdown("""
    <style>
    body {
            
        background-color: #0e0e1a;
    }
    .stApp {
        background-color: #0e0e1a;
        color: white;
    }
    .chat-user {
        background-color: #6a0dad;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .chat-bot {
        background-color: #ffb6c1;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 NER Chatbot")

# -----------------------------
# Load model
# -----------------------------
try:
    with open("intent_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    st.success("Model loaded successfully ✅")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -----------------------------
# Preprocessing
# -----------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w.isalnum() and w not in stop_words]
    return " ".join(words)

# -----------------------------
# NER
# -----------------------------
ignore_words = ["find", "in", "names", "locations", "organizations"]
locations_list = ["Lahore", "Karachi", "Islamabad", "London", "Paris","China","India","America","Rawalpindi"]

def extract_entities_from_text(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)

    entities = {
        "Person": [],
        "Location": [],
        "Organization": [],
        "Date": []
    }

    # simple date words
    dates_list = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April"
    ]

    for word, tag in tagged:

        if word.lower() in ignore_words:
            continue

        if word in locations_list:
            entities["Location"].append(word)

        elif word in dates_list:
            entities["Date"].append(word)

        elif tag == "NNP":
            entities["Person"].append(word)

    return entities

# -----------------------------
# Chatbot
# -----------------------------
def chatbot(user_input):

    processed = preprocess(user_input)
    vec = vectorizer.transform([processed])
    intent = model.predict(vec)[0]

    entities = extract_entities_from_text(user_input)

    # ---------- Mode 1: specific intent ----------
    if "find" in user_input.lower():

        if intent == "find_person":
            return "👤 Persons: " + ", ".join(entities["Person"]) if entities["Person"] else "👤 Persons: Empty"

        elif intent == "find_location":
            return "📍 Locations: " + ", ".join(entities["Location"]) if entities["Location"] else "📍 Locations: Empty"

        elif intent == "find_organization":
            return "🏢 Organizations: " + ", ".join(entities["Organization"]) if entities["Organization"] else "🏢 Organizations: Empty"

        elif intent == "find_date":
            return "📅 Dates: " + ", ".join(entities["Date"]) if entities["Date"] else "📅 Dates: Empty"

    # ---------- Mode 2: general NER ----------
    else:
        response = f"""
👤 Person: {", ".join(entities["Person"]) if entities["Person"] else "Empty"}

📍 Location: {", ".join(entities["Location"]) if entities["Location"] else "Empty"}

📅 Date: {", ".join(entities["Date"]) if entities["Date"] else "Empty"}

🏢 Organization: {", ".join(entities["Organization"]) if entities["Organization"] else "Empty"}
"""
        return response
  

# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

# Input
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Bot response
    response = chatbot(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
