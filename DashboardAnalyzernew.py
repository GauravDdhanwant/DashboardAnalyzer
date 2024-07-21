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
from yolo import YOLO  # Assume you have a YOLO implementation or use a pre-trained model

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

# Load YOLO model
yolo = YOLO(model_path='yolo.h5', classes_path='yolo_classes.txt')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def extract_text(image):
    results = reader.readtext(image, detail=0)
    text = ' '.join(results)
    return text

def detect_visuals(image):
    visuals = []
    results = yolo.detect_image(image)
    for box in results:
        x, y, w, h, class_name = box
        if w > 50 and h > 50:  # Filter out small regions
            visuals.append((image[y:y+h, x:x+w], class_name))
    return visuals

def analyze_visual(visual, class_name):
    if class_name in ['chart', 'table']:
        processed_visual = preprocess_image(visual)
        text = extract_text(processed_visual)
        summary, action_items = generate_insights_and_actions_from_gpt(text)
        return summary, action_items
    else:
        return "Unsupported visual type", "No action items available"

def generate_insights_and_actions_from_gpt(text):
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
                {"role": "system", "content": "You are a helpful assistant for summarizing business dashboards and providing a clear understanding of the numerical insights of the information presented visual by visual by understanding what is the purpose of the visual."},
                {"role": "user", "content": detailed_prompt}
            ]
        )
        if response and response.choices and len(response.choices) > 0:
            result = response.choices[0].message['content'].strip()
            if "Summary:" in result and "Action Items:" in result:
                summary, action_items = result.split("Action Items:")
                summary = summary.replace("Summary:", "").strip()
                action_items = action_items.strip()
                return summary, action_items
            else:
                return result, "No specific action items found."
        else:
            return "No response from OpenAI.", "No specific action items found."
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

if uploaded_file is not None and openai.api_key:
    # Read the uploaded file using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_np = np.array(image)
    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption='Uploaded Screenshot', use_column_width=True)

    visuals = detect_visuals(image_np)
    analysis_results = [analyze_visual(visual, class_name) for visual, class_name in visuals]

    st.header("Analysis")

    for i, (summary, action_items) in enumerate(analysis_results):
        if summary and action_items:
            st.subheader(f"Summary for Visual {i+1}")
            st.write(summary)
            st.subheader(f"Action Items for Visual {i+1}")
            st.write(action_items)
        else:
            st.write(f"Error analyzing visual {i+1}")

