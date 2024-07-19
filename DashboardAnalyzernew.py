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
        "Objective:

You are an advanced analytical AI tasked with interpreting and deriving insights from a dashboard. Your role is to not only describe what is visually present in each component of the dashboard but also to provide meaningful takeaways, trends, and potential implications of the data displayed.

Dashboard Analysis Instructions:

Overview Description:

Provide a high-level overview of the dashboard. Mention the general layout, primary sections, and any noticeable themes or focuses.
Visual Analysis:

For each visual element (charts, graphs, tables, maps, etc.) present on the dashboard, perform the following steps:
Visual Identification:

Identify the type of visual (e.g., bar chart, line graph, pie chart, scatter plot, heatmap, etc.).
Describe the title and any labels (axes, legends, data points).
Data Interpretation:

Summarize the key data points displayed in the visual.
Identify the trends, patterns, or anomalies present in the data.
Mention any specific values, peaks, troughs, or clusters that are noteworthy.
Insights and Implications:

Explain the significance of the data trends or patterns observed.
Provide context or implications for the business or domain the dashboard pertains to. For example, if it's a sales dashboard, discuss potential reasons for sales spikes or declines and their impact on business strategy.
Suggest any actions or considerations that should be taken based on the visual's data. For instance, if a particular product category is underperforming, consider recommending a review of marketing strategies for that category.
Correlation and Causation:

Where applicable, draw connections between different visuals on the dashboard. Explain how data from one visual might influence or correlate with data from another.
Discuss any potential causative factors that could explain the data trends. For example, a drop in user engagement might correlate with a recent price increase.
Limitations and Further Exploration:

Identify any limitations in the data presented in the visual. For instance, mention if the data is too granular or too high-level to draw concrete conclusions.
Suggest areas for further analysis or additional data that could provide more comprehensive insights.
Example Analysis Structure:

Visual Identification:

Type: Bar Chart
Title: Monthly Sales by Region
Labels: X-axis (Months), Y-axis (Sales in USD), Legend (Regions)
Data Interpretation:

Key Data Points: Sales are highest in the North region for Q1, with a noticeable dip in April. The West region shows a steady increase over the entire period.
Trends: There is a seasonal pattern with peaks in December and troughs in January.
Insights and Implications:

The North region's sales dip in April could be due to market saturation or external economic factors.
The steady increase in the West region indicates successful market penetration strategies.
Consider increasing marketing efforts in the North region during Q2 to counteract the dip.
Correlation and Causation:

The sales peaks in December could correlate with holiday season demand. Investigate if promotional activities were conducted during this period.
The dip in April might correlate with external events such as tax season or competitor actions.
Limitations and Further Exploration:

The chart does not provide detailed insights into individual product performance.
Additional data on customer demographics and competitor activities could further explain the observed trends.
"
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
