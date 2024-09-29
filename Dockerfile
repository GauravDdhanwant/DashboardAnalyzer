# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies and Chrome
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    gnupg \
    ca-certificates \
    --no-install-recommends

# Add Google Chrome signing key and repository
RUN curl -fsSL https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable

# Install ChromeDriver based on Chrome version
RUN CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+') \
    && wget -q "https://chromedriver.storage.googleapis.com/${CHROME_VERSION}/chromedriver_linux64.zip" -O /tmp/chromedriver.zip \
    && unzip /tmp/chromedriver.zip -d /usr/local/bin/ \
    && rm /tmp/chromedriver.zip

# Set environment variables for Chrome and ChromeDriver
ENV CHROME_BIN=/usr/bin/google-chrome
ENV CHROME_PATH=/usr/local/bin/chromedriver

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
