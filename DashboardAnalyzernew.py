import streamlit as st
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import google.generativeai as genai
import requests
from io import BytesIO
from PIL import Image
import time

# Function to configure the API key for Google Generative AI
def configure_api(api_key):
    st.write("Configuring Google Generative AI API key...")
    try:
        genai.configure(api_key=api_key)
        st.success("API key configured successfully!")
    except Exception as e:
        st.error(f"Failed to configure API key: {e}")

# Function to automatically install ChromeDriver
def install_chromedriver():
    st.write("Installing ChromeDriver with chromedriver-autoinstaller...")
    try:
        chromedriver_autoinstaller.install()
        st.success("ChromeDriver installed successfully!")
    except Exception as e:
        st.error(f"Failed to install ChromeDriver: {e}")

# Function to take a screenshot of a dashboard from a URL
def take_screenshot(url):
    st.write("Taking a screenshot of the dashboard...")
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        # Initialize WebDriver with the installed ChromeDriver
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(10)  # Allow the page to fully load
        
        screenshot = driver.get_screenshot_as_png()
        driver.quit()
        st.success("Screenshot taken successfully!")
        return Image.open(BytesIO(screenshot))
    except Exception as e:
        st.error(f"Failed to take a screenshot: {e}")
        return None

# Function to extract HTML content from the dashboard URL
def extract_dashboard_content(url):
    st.write("Extracting HTML content from the dashboard...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            st.success("HTML content extracted successfully!")
            return soup
        else:
            st.error(f"Failed to retrieve the dashboard. HTTP status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Failed to extract HTML content: {e}")
        return None

# Function to generate insights from the extracted content
def generate_insights(soup, question=None):
    st.write("Generating insights using Google Generative AI...")
    try:
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
        st.success("Insights generated successfully!")
        return response.text
    except Exception as e:
        st.error(f"Failed to generate insights: {e}")
        return None

# Function to handle user questions and generate insights
def handle_conversation(soup):
    st.subheader("Ask your questions about the dashboard")
    question = st.text_input("Your question:")
    
    if st.button("Ask"):
        with st.spinner("Analyzing the dashboard..."):
            result = generate_insights(soup, question)
            if result:
                st.write(result)
            else:
                st.error("Failed to generate insights. Please try again.")

# Streamlit app setup
st.set_page_config(page_title="Dashboard Analyzer", page_icon=":bar_chart:", layout="wide")

# Sidebar for user input
st.sidebar.title("Dashboard Analyzer")
api_key = st.sidebar.text_input("Enter your API Key", type="password")
dashboard_url = st.sidebar.text_input("Enter Dashboard URL")

# Install ChromeDriver automatically
install_chromedriver()

# Analyze the dashboard when the button is clicked
if st.sidebar.button("Analyze"):
    if api_key and dashboard_url:
        configure_api(api_key)
        
        with st.spinner("Processing..."):
            soup = extract_dashboard_content(dashboard_url)
            
            if soup:
                st.subheader("Dashboard Preview")
                screenshot = take_screenshot(dashboard_url)
                if screenshot:
                    st.image(screenshot, caption="Dashboard Screenshot", use_column_width=True)

                st.subheader("Dashboard Content")
                st.text(soup.prettify()[:1000])  # Display a part of the HTML content

                handle_conversation(soup)
            else:
                st.error("Could not extract the dashboard content.")
    else:
        st.warning("Please enter both the API Key and the Dashboard URL.")
