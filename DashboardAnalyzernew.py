import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import time

# Function to configure the API key
def configure_api(api_key):
    genai.configure(api_key=api_key)

# Function to take a screenshot of the dashboard
def take_screenshot(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_service = Service('chromedriver.exe')  # Update this path
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    
    driver.get(url)
    time.sleep(5)  # Allow the page to fully load
    
    screenshot = driver.get_screenshot_as_png()
    driver.quit()
    
    return Image.open(BytesIO(screenshot))

# Function to extract HTML content from the dashboard
def extract_dashboard_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        return soup
    else:
        st.error("Failed to retrieve the dashboard.")
        return None

# Function to generate insights from the extracted content
def generate_insights(soup, question=None):
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 4096,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=generation_config)

    input_prompt = "You are an expert in reading and analyzing dashboards."

    if question:
        question_prompt = f"Question: {question}"
    else:
        question_prompt = "Provide a comprehensive summary and insights based on the dashboard content."

    dashboard_content = str(soup)
    prompt = [input_prompt, dashboard_content, question_prompt]

    response = model.generate_content(prompt)
    return response.text

# Function to handle the conversation
def handle_conversation(soup):
    st.subheader("Ask your questions about the dashboard")
    question = st.text_input("Your question:")
    
    if st.button("Ask"):
        with st.spinner("Analyzing the dashboard..."):
            answer = generate_insights(soup, question)
            st.write(answer)

# Streamlit app
st.set_page_config(page_title="Dashboard Analyzer", page_icon=":bar_chart:", layout="wide")

# Apply the theme
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #000000;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: #ffffff;
        border-radius: 4px;
        border: none;
    }
    .stImage {
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("Dashboard Analyzer")

api_key = st.sidebar.text_input("Enter your API Key", type="password")
dashboard_url = st.sidebar.text_input("Enter Dashboard URL")

if api_key and dashboard_url:
    configure_api(api_key)
    
    with st.spinner("Extracting dashboard content..."):
        soup = extract_dashboard_content(dashboard_url)
    
    if soup:
        st.subheader("Dashboard Preview")
        screenshot = take_screenshot(dashboard_url)
        st.image(screenshot, caption="Dashboard Screenshot", use_column_width=True)
        
        st.subheader("Generated Insights")
        insights = generate_insights(soup)
        st.write(insights)
        
        handle_conversation(soup)
else:
    st.warning("Please enter your API Key and Dashboard URL.")
