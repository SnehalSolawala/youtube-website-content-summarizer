# youtube-website-content-summarizer
tract and summarize text from YouTube videos and websites using LangChain and ChatGroq LLM.
# LangChain Text Summarizer for YouTube & Websites

This Streamlit app extracts text from YouTube videos or websites and generates a concise 300-word summary using LangChain and ChatGroq.

## Features

- Extracts text from YouTube videos using **yt-dlp** and **Whisper transcription**.
- Extracts website content using:
  - **Static HTML scraping** with BeautifulSoup
  - **Dynamic content scraping** with Playwright
- Summarizes text using **ChatGroq LLM** with a structured prompt template.
- Displays **input text** and **summary** separately in Streamlit with distinct colors.
- Handles exceptions and invalid URLs gracefully.

## Requirements

- Python 3.10+
- Streamlit
- LangChain
- LangChain Community
- ChatGroq
- yt-dlp
- OpenAI Whisper
- Playwright
- Requests
- BeautifulSoup4
- Validators

## Installation

```bash
pip install streamlit langchain langchain_community playwright requests beautifulsoup4 validators
playwright install 
```

## Usage

1. Run the Streamlit app:
        streamlit run app.py

2. Enter your Groq API Key in the sidebar.

3. Paste a YouTube video URL or a website URL.

4. Click Summarize the content from YouTube or website.

5. View the input text and summary in separate sections.
