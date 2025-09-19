import os
import subprocess
import validators
import streamlit as st
import warnings
from urllib3.exceptions import InsecureRequestWarning
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from playwright.async_api import async_playwright
import asyncio
import requests
from bs4 import BeautifulSoup

# Ignore SSL warnings
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# -------------------------
# YouTube transcription
# -------------------------
import yt_dlp
import whisper
def extract_youtube_text(url, lang="en", model="tiny"):
    base_file = "audio"   # no extension!
    audio_file = f"{base_file}.mp3"
    transcript_file = "transcription.txt"

    # Clean old files
    if os.path.exists(audio_file):
        os.remove(audio_file)
    if os.path.exists(transcript_file):
        os.remove(transcript_file)

    # yt-dlp config
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": base_file,   # no extension
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Now audio.mp3 will exist
    st.info(f"🎙 Running Whisper transcription with '{model}' model...")
    whisper_model = whisper.load_model(model)
    result = whisper_model.transcribe(audio_file, language=lang)

    text_content = result["text"]

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(text_content)

    os.remove(audio_file)

    st.success(f"✅ Transcription completed. Saved to '{transcript_file}'")
    return text_content


# -------------------------
# Dynamic website text extraction (hybrid)
# -------------------------
async def fetch_dynamic_text(url):
    """Hybrid extraction: static + Playwright fallback"""
    # --- Static extraction first ---
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            main = soup.find("article") or soup.find(class_="content") or soup.find(class_="main")
            if main:
                text = main.get_text(separator=" ", strip=True)
                if len(text) > 200:
                    return text.encode("utf-8", errors="ignore").decode("utf-8")
            return soup.get_text(separator=" ", strip=True).encode("utf-8", errors="ignore").decode("utf-8")
    except Exception as e:
        print(f"Static extraction failed: {e}")

    # --- Playwright dynamic extraction ---
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            await page.wait_for_load_state("networkidle")

            # Scroll for lazy-loaded content
            prev_height = None
            while True:
                cur_height = await page.evaluate("document.body.scrollHeight")
                if prev_height == cur_height:
                    break
                prev_height = cur_height
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)

            # Try multiple selectors
            selectors = ["article", ".content", ".main", "body"]
            text = ""
            for sel in selectors:
                try:
                    text = await page.inner_text(sel)
                    if text.strip():
                        break
                except Exception:
                    continue

            # Fallback to entire HTML
            if not text.strip():
                html = await page.content()
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)

            await browser.close()
            return text.encode("utf-8", errors="ignore").decode("utf-8")

    except Exception as e:
        print(f"Playwright extraction failed: {e}")
        return ""

# -------------------------
# Streamlit app
# -------------------------
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Groq API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Initialize LLM
if groq_api_key:
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
else:
    st.error("Please enter a valid Groq API Key in the sidebar.")

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

if st.button("Summarize the content from YouTube or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YT or website)")
    else:
        try:
            with st.spinner("Extracting content..."):
                docs = []

                # Case 1: YouTube URL
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        text_content = extract_youtube_text(generic_url, model="tiny")
                        docs = [Document(page_content=text_content)]
                    except Exception as yt_error:
                        st.error(f"Could not fetch transcript: {yt_error}")

                # Case 2: Website URL
                else:
                    # Static extraction first
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        docs = loader.load()
                    except Exception as e:
                        st.warning(f"UnstructuredURLLoader failed: {e}")

                    # Playwright fallback
                    if not docs or not docs[0].page_content.strip():
                        try:
                            text_content = asyncio.run(fetch_dynamic_text(generic_url))
                            docs = [Document(page_content=text_content)]
                        except Exception as e:
                            st.error(f"Playwright extraction failed: {e}")

                if not docs or not docs[0].page_content.strip():
                    st.error("Failed to extract text from this URL.")
                else:
                    # Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.invoke(docs)
                    # Display input text
                    st.success(f"Input Text: {output_summary['input_documents'][0].page_content}")  # truncated for readability

                    # Display summary
                    st.success(f"Summary: {output_summary['output_text']}")
                    
            
        except Exception as e:
                st.exception(f"Exception: {e}")    
