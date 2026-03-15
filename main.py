import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
PROJECT_ID = "transcript-summarizer-490013"
LOCATION = "asia-southeast1"

# Required for NLTK to work on Cloud instances
@st.cache_resource
def load_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

load_nltk()

# --- 2. AUTHENTICATION ---
# Streamlit Cloud uses st.secrets to securely pass GCP credentials
if "gcp_service_account" in st.secrets:
    creds_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
else:
    st.error("GCP Credentials not found in Streamlit Secrets!")
    st.stop()

model = GenerativeModel(
    "gemini-2.0-flash",
    system_instruction="Keep every response extremely brief. Summaries must be 5 short bullets. All follow-ups under 50 words."
)

# --- 3. SESSION STATE ---
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""
if "input_text_val" not in st.session_state:
    st.session_state.input_text_val = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- 4. LOGIC FUNCTIONS ---
def clean_text(text):
    if not text: return ""
    text = re.sub(r'[^a-zA-Z0-9\s\.\?\!]', '', text).lower()
    stop_words = set(stopwords.words('english'))
    fillers = {'um', 'uh', 'ah', 'er', 'basically', 'actually', 'you know', 'sort of', 'like'}
    words = text.split()
    return " ".join([w for w in words if w not in stop_words and w not in fillers]).strip()

def extract_data(files_list):
    combined_text = ""
    for f in files_list:
        if f.name.endswith('.txt'):
            combined_text += f.read().decode("utf-8") + "\n"
        elif f.name.endswith('.xlsx'):
            df = pd.read_excel(f)
            combined_text += " ".join(df.astype(str).values.flatten()) + "\n"
    return combined_text

def get_overlapping_chunks(text, chunk_size=5000, overlap=500):
    chunks = []
    if not text: return chunks
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        if start >= len(text): break
    return chunks

# --- 5. UI ---
st.set_page_config(page_title="AI Transcript Analyser", layout="wide")
st.title("Transcript Analysis Dashboard")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Input")
    has_text = len(st.session_state.input_text_val.strip()) > 0
    
    uploaded_files = st.file_uploader(
        "Upload TXT/XLSX", 
        type=["txt", "xlsx"], 
        accept_multiple_files=True,
        disabled=has_text,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    manual_input = st.text_area(
        "Paste Transcript:", 
        height=300, 
        value=st.session_state.input_text_val,
        disabled=bool(uploaded_files)
    )
    st.session_state.input_text_val = manual_input
    
    if st.button("🚀 Generate", use_container_width=True):
        raw_data = extract_data(uploaded_files) if uploaded_files else manual_input
        if raw_data:
            cleaned = clean_text(raw_data)
            chunks = get_overlapping_chunks(cleaned)
            
            try:
                if len(chunks) > 1:
                    chunk_summaries = [model.generate_content(f"Summarize briefly: {c}").text for c in chunks]
                    final_prompt = f"Combine these into 5 points:\n\n{' '.join(chunk_summaries)}"
                else:
                    final_prompt = f"Summarize in 5 short points:\n\n{cleaned}"

                st.session_state.chat_session = model.start_chat()
                response = st.session_state.chat_session.send_message(final_prompt)
                st.session_state.summary_text = response.text
                st.session_state.messages = [{"role": "assistant", "content": response.text}]
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.update({"input_text_val": "", "summary_text": "", "messages": []})
        st.session_state.uploader_key += 1
        st.rerun()

with col2:
    st.subheader("Summary & Chat")
    with st.container(height=450):
        if not st.session_state.messages:
            st.write("Results will appear here.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Follow-up..."):
        if st.session_state.summary_text:
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.chat_session.send_message(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()

    if st.session_state.summary_text:
        st.download_button("💾 Download Summary", st.session_state.summary_text, "summary.txt")
