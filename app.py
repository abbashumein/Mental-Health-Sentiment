
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Title
st.title("🧠 Mental Health Sentiment & Severity Detector")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=28)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# User input
text = st.text_area("Enter your mental health-related text here:")

# Predict
if st.button("Analyze"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze()

    # Display results
    threshold = 0.5
    emotion_labels = [f"Label {i}" for i in range(28)]  # replace with actual emotion names if available
    detected = [(emotion_labels[i], float(probs[i])) for i in range(len(probs)) if probs[i] > threshold]

    if detected:
        st.subheader("Detected Emotions:")
        for emotion, score in detected:
            st.write(f"- {emotion} (Confidence: {score:.2f})")
    else:
        st.write("No strong emotion detected.")
