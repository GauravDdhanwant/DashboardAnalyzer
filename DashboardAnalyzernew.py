import streamlit as st
import numpy as np
import easyocr
import openai
import cv2
from bidi.algorithm import get_display

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

def detect_visual_elements(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    visual_elements_img = image.copy()
    cv2.drawContours(visual_elements_img, contours, -1, (0, 255, 0), 3)

    return visual_elements_img, contours

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

    # Use OpenCV to extract visual elements
    visual_elements_img, contours = detect_visual_elements(screenshot)

    st.image(cv2.cvtColor(visual_elements_img, cv2.COLOR_BGR2RGB), caption='Detected Visual Elements', use_column_width=True)

    # Use GPT-4 to generate a human-readable summary of the extracted text and visual elements
    summary = generate_summary_from_gpt(text, contours)

    analysis_result['summary'] = summary

    return analysis_result

def generate_summary_from_gpt(text, visual_elements):
    detailed_prompt = (
        "You are a helpful assistant for summarizing business dashboards and providing a clear understanding of the information presented. "
        "Read the visualizations and draw meaningful insights. Here is the extracted text from a business dashboard: \n\n"
        f"{text}\n\n"
        "Based on this information and the detected visual elements, provide a clear and concise summary that explains what is present in the image in an easy-to-understand form. "
        "Consider the types of charts (e.g., bar charts, line charts, pie charts), key metrics, and any notable trends or patterns."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for summarizing business dashboards and providing a clear understanding of numerical insights of the information presented."},
                {"role": "user", "content": detailed_prompt}
            ]
        )
        summary = response['choices'][0]['message']['content'].strip()
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
