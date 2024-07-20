#!/usr/bin/env python
# coding: utf-8

# In[3]:
import streamlit as st
import numpy as np
import easyocr
import openai
import cv2
from PIL import Image
import pytesseract

st.set_page_config(layout="wide")

# Set up OpenAI API key
openai_api_key_input = st.text_input("Enter OpenAI API Key", type="password")
openai.api_key = openai_api_key_input

if not openai.api_key:
    st.error("OpenAI API key is not set. Please set it as an environment variable 'OPENAI_API_KEY'.")

# Streamlit UI
st.sidebar.title("Dashboard Analyzer")

uploaded_file = st.sidebar.file_uploader("Upload a Screenshot", type=["png", "jpg", "jpeg"])

# Initialize EasyOCR reader with a progress bar
with st.spinner("Downloading OCR model... This may take a few minutes."):
    reader = easyocr.Reader(['en'])

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def extract_text(image):
    results = reader.readtext(image, detail=0)
    text = ' '.join(results)
    return text

def detect_visuals(image):
    # Placeholder for visual detection logic
    # For simplicity, we'll assume the entire image is one visual in this example
    visuals = [image]
    return visuals

def analyze_visual(visual):
    processed_visual = preprocess_image(visual)
    text = extract_text(processed_visual)
    summary = generate_summary_from_gpt(text)
    return summary

def generate_summary_from_gpt(text):
    detailed_prompt = (
        "You are a helpful assistant for summarizing business dashboards and providing a clear understanding of the information presented. Read the visualizations and draw meaningful insights. "
        "Here is the extracted text from a customer service team quality assessment dashboard: \n\n"
        f"{text}\n\n"
        "Based on this information, provide a clear and concise summary that explains what is present in the image in an easy-to-understand form. "
        "Additionally, provide actionable recommendations based on the data."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for summarizing business dashboards and providing a clear understanding numerical insights of the information presented visual by visual by understanding what is the purpose of the visual"},
                {"role": "user", "content": detailed_prompt}
            ]
        )
        summary = response.choices[0].message['content'].strip()
        return summary
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

if uploaded_file is not None and openai.api_key:
    # Read the uploaded file using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_np = np.array(image)
    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption='Uploaded Screenshot', use_column_width=True)

    visuals = detect_visuals(image_np)
    analysis_results = [analyze_visual(visual) for visual in visuals]

    st.header("Analysis")

    for i, result in enumerate(analysis_results):
        st.subheader(f"Summary for Visual {i+1}")
        st.write(result)
