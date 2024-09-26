import streamlit as st
from bs4 import BeautifulSoup
import google.generativeai as genai

# Function to configure the API key
def configure_api(api_key):
    genai.configure(api_key=api_key)

# Load HTML data from the uploaded file
def load_html_file(uploaded_file):
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            soup = BeautifulSoup(content, "html.parser")
            return soup
        except Exception as e:
            st.error(f"Error parsing HTML: {e}")
            return None
    else:
        return None

# Insight generation from HTML data
def generate_insights_from_html(soup, question, task_type):
    html_prompts = [str(soup)]  # Modify as needed to extract relevant parts
    return get_image_info(html_prompts, question, task_type)

# Task type identification based on HTML data
def identify_task_type(html_prompts, question):
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 4096,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

    input_prompt = """You are an expert in reading and analyzing the charts."""

    question_prompt = f"""Given a chart and a question, you have to tell which category the question belongs to. Only return the category type of the question and nothing else.
                          Question : {question}

                          Categories: 
                          1. If the question is related to question answering or numerical question answering based on chart return 'Question Answering'
                          2. If the question is related to Chart Summarization or Chart Analysis return 'Summarization'
                          3. If there are multiple images/charts provided, return 'Comparison'
                      """

    prompt_parts = [input_prompt] + html_prompts + [question_prompt]
    response = model.generate_content(prompt_parts)
    return str(response.text).strip()

# Function to generate insights
def get_image_info(image_prompts, question, task_type):
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 4096,
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=generation_config)

    input_prompt = """You are an expert in reading and analyzing the charts."""

    question_prompt = f"""Question : {question}"""

    # Adjust this logic based on task_type
    if task_type == "Summarization":
        # Add specific logic for summarization
        pass
    elif task_type == "Question Answering":
        # Add specific logic for question answering
        pass
    elif task_type == "Comparison":
        # Add specific logic for comparison
        pass

    prompt_parts = [input_prompt] + image_prompts + [question_prompt]
    response = model.generate_content(prompt_parts)
    return str(response.text)

# Streamlit app
st.set_page_config(page_title="InsightsBoard", page_icon=":bar_chart:", layout="wide")

# Apply the theme
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #000000;  /* Set sidebar background to black */
    }
    .sidebar .sidebar-content .sidebar-header {
        background-color: #1e3a8a;
        color: #ffffff;
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

st.sidebar.title("InsightsBoard")

api_key = st.sidebar.text_input("Enter your API Key", type="password")

if api_key:
    configure_api(api_key)

    uploaded_file = st.sidebar.file_uploader("Upload HTML Dashboard File", type=["txt"])
    question = st.sidebar.text_input("Enter Your Question Here")

    if uploaded_file and question:
        with st.spinner("Processing..."):
            soup = load_html_file(uploaded_file)
            if soup:
                task_type = identify_task_type([str(soup)], question)
                result = generate_insights_from_html(soup, question, task_type)
                st.write(result)
            else:
                st.warning("Please upload a valid HTML file.")
else:
    st.warning("Please enter your API Key.")
