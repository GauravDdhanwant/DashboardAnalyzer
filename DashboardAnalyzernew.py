#!/usr/bin/env python
# coding: utf-8

# In[3]:
import streamlit as st
import numpy as np
import easyocr
import openai
import os
import cv2

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

    # Use EasyOCR to extract text
    results = reader.readtext(gray_screenshot, detail=0)
    text = ' '.join(results)

    # Use GPT-4 to generate a human-readable summary of the extracted text
    summary = generate_summary_from_gpt(text)

    analysis_result['summary'] = summary

    return analysis_result

def generate_summary_from_gpt(text):
    detailed_prompt = (
    "You are an advanced analytical AI tasked with interpreting and deriving insights from a dashboard. Your role is to not only describe what is visually present in each component of the dashboard but also to provide meaningful takeaways, trends, and potential implications of the data displayed.Start with a high-level overview of the dashboard, mentioning the general layout, primary sections, and any noticeable themes or focuses. Describe the structure of the dashboard, noting the arrangement of visuals and any prominent features that stand out.For each visual element on the dashboard, identify the type of visual (e.g., bar chart, line graph, pie chart, scatter plot, heatmap, etc.) and provide a detailed description of the title and any labels (axes, legends, data points). Summarize the key data points displayed in the visual, highlighting the main figures and any significant values. Identify the trends, patterns, or anomalies present in the data, mentioning any specific peaks, troughs, or clusters that are noteworthy.Explain the significance of the data trends or patterns observed in the visual, providing context or implications for the business or domain the dashboard pertains to. For example, if it’s a sales dashboard, discuss potential reasons for sales spikes or declines and their impact on business strategy. Suggest any actions or considerations that should be taken based on the visual’s data, such as reviewing marketing strategies for underperforming product categories.Where applicable, draw connections between different visuals on the dashboard, explaining how data from one visual might influence or correlate with data from another. Discuss any potential causative factors that could explain the data trends. For instance, a drop in user engagement might correlate with a recent price increase.Identify any limitations in the data presented in the visual, mentioning if the data is too granular or too high-level to draw concrete conclusions. Suggest areas for further analysis or additional data that could provide more comprehensive insights.For example, consider a bar chart titled Monthly Sales by Region, with the X-axis labeled as Months and the Y-axis labeled as Sales in USD, along with a legend indicating different regions. Summarize that the sales are highest in the North region for Q1, with a noticeable dip in April, and the West region shows a steady increase over the entire period. Highlight the seasonal pattern with peaks in December and troughs in January. Explain that the North region's sales dip in April could be due to market saturation or external economic factors, while the steady increase in the West region indicates successful market penetration strategies. Recommend increasing marketing efforts in the North region during Q2 to counteract the dip. Draw a connection between the sales peaks in December and holiday season demand, suggesting further investigation into promotional activities during this period. Mention that the chart does not provide detailed insights into individual product performance and suggest additional data on customer demographics and competitor activities to further explain the observed trends.Always ensure your analysis is thorough and provides actionable insights beyond the surface-level data. The goal is to enable stakeholders to make informed decisions based on your interpretations."

    "Here is the extracted text from a customer service team quality assessment dashboard: \n\n"
f"{text}\n\n"
"Based on this information, provide a clear and concise summary that explains what is present in the image in an easy-to-understand form."
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

    analysis_result = analyze_screenshot(image_np)

    st.header("Analysis")

    st.subheader("Summary")
    st.write(analysis_result['summary'])
