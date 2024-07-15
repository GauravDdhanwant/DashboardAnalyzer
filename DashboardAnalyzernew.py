#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
import pytesseract
import openai
import os
import cv2
st.set_page_config(layout="wide")
# Ensure Tesseract OCR is available
# tesseract_installed = False
# try:
#     pytesseract.get_tesseract_version()
#     tesseract_installed = True
# except pytesseract.pytesseract.TesseractNotFoundError:
#     st.error("Tesseract OCR is not installed or not found in PATH. Please install it following the instructions provided [here](https://github.com/tesseract-ocr/tesseract/wiki).")

# Set up OpenAI API key
openai.api_key = st.text_input("Enter OpenAI API Key", type="password")

if not openai.api_key:
    st.error("OpenAI API key is not set. Please set it as an environment variable 'OPENAI_API_KEY'.")

def analyze_screenshot(screenshot):
    analysis_result = {}

    # Check the number of channels in the image
    st.write(f"Image shape: {screenshot.shape}")

    if len(screenshot.shape) == 2:  # Grayscale image
        gray_screenshot = screenshot
    elif len(screenshot.shape) == 3:
        if screenshot.shape[2] == 4:  # 4 channels (BGRA)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    else:
        st.error("Unexpected image format.")
        return

    # Use OCR to extract text
    text = pytesseract.image_to_string(gray_screenshot)

    # Use GPT-4 to generate a human-readable summary of the extracted text
    summary = generate_summary_from_gpt(text)

    analysis_result['summary'] = summary

    return analysis_result

def generate_summary_from_gpt(text):
    detailed_prompt = (
        "You are a helpful assistant for summarizing business dashboards and providing a clear understanding of the information presented. "
        "Here is the extracted text from a customer service team quality assessment dashboard: \n\n"
        f"{text}\n\n"
        "Based on this information, provide a clear and concise summary that explains what is present in the image in an easy-to-understand form."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for summarizing business dashboards and providing a clear understanding of the information presented."},
            {"role": "user", "content": detailed_prompt}
        ]
    )

    summary = response.choices[0].message['content'].strip()

    return summary

# Streamlit UI
st.sidebar.title("Dashboard Analyzer")

uploaded_file = st.sidebar.file_uploader("Upload a Screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and openai.api_key:
    # Read the uploaded file using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_np = np.array(image)
    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption='Uploaded Screenshot', use_column_width=True)

    analysis_result = analyze_screenshot(image_np)

    st.header("Analysis")

    st.subheader("Summary")
    st.write(analysis_result['summary'])
