import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io
import json 
import asyncio 
import aiohttp 
import nest_asyncio 
import time 

nest_asyncio.apply()


CONCURRENT_LLM_CALLS_LIMIT = 5 
MAX_RETRIES = 3 
INITIAL_RETRY_DELAY = 5 

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: "
             "`python -m spacy download en_core_web_sm` in your terminal.")
    st.stop() 

def extract_text_from_pdf(pdf_file_stream):
    """
    Extracts text from a PDF file stream.
    """
    try:
        reader = PyPDF2.PdfReader(pdf_file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or "" 
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def preprocess_text(text, nlp_model):
    """
    Preprocesses text by tokenizing, lemmatizing, and removing stop words and punctuation.
    Returns a list of significant tokens.
    """
    doc = nlp_model(text.lower())

    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
    return " ".join(tokens)

def extract_significant_keywords(text, nlp_model):
    """
    Extracts potential keywords/phrases using noun chunks and named entities.
    Prioritizes longer phrases and entities.
    """
    doc = nlp_model(text.lower())
    keywords = set()

    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.strip())

    for ent in doc.ents:
        keywords.add(ent.text.strip())

    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            keywords.add(token.lemma_)

    generic_terms = {"experience", "skills", "management", "strong", "proven", "ability", "key", "developer", "engineer", "analyst", "specialist", "role"}
    return {kw for kw in keywords if len(kw) > 1 and kw not in generic_terms}


def calculate_match_and_keywords(job_desc_text, resume_text, nlp_model):
    """
    Calculates the cosine similarity between job description and resume,
    and identifies matching/missing keywords.
    """
    if not job_desc_text or not resume_text:
        return 0, set(), set(), set()

    processed_job_desc = preprocess_text(job_desc_text, nlp_model)
    processed_resume = preprocess_text(resume_text, nlp_model)

    vectorizer = TfidfVectorizer().fit([processed_job_desc, processed_resume])
    job_desc_vector = vectorizer.transform([processed_job_desc])
    resume_vector = vectorizer.transform([processed_resume])

    similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0]
    match_percentage = round(similarity * 100, 2)

    job_keywords = extract_significant_keywords(job_desc_text, nlp_model)
    resume_keywords = extract_significant_keywords(resume_text, nlp_model)

    matching_keywords = job_keywords.intersection(resume_keywords)
    missing_keywords = job_keywords.difference(resume_keywords)
    extra_resume_keywords = resume_keywords.difference(job_keywords)

    return match_percentage, list(matching_keywords), list(missing_keywords), list(extra_resume_keywords)

llm_semaphore = asyncio.Semaphore(CONCURRENT_LLM_CALLS_LIMIT)

async def get_resume_enhancement_suggestion(missing_keyword, job_description_snippet):
    """
    Uses the Gemini API to get a concise suggestion for a missing keyword,
    respecting a concurrency limit and implementing retry logic for 429 errors.
    """
 
    async with llm_semaphore:
        prompt = f"""You are a resume enhancement assistant. For the following job description snippet and a missing keyword, provide a single, concise, and actionable suggestion (max 2 sentences) on how a candidate can add context to their resume to include this skill/keyword. Focus on real-world examples, projects, or quantifiable achievements.

        Job Description Snippet (relevant parts, capped at 500 characters to avoid exceeding token limits):
        ---
        {job_description_snippet[:500]}
        ---

        Missing Keyword: {missing_keyword}

        Suggestion:"""

        if "GEMINI_API_KEY" in st.secrets:
            apiKey = st.secrets["GEMINI_API_KEY"]
        else:
            st.error("Gemini API key not found in Streamlit secrets. "
                     "Please add `GEMINI_API_KEY = 'YOUR_API_KEY'` to your `.streamlit/secrets.toml` file.")
            return "API key not configured." 

        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"


        chatHistory = []
        chatHistory.append({ "role": "user", "parts": [{ "text": prompt }] })
        payload = { "contents": chatHistory }

        for attempt in range(MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload)) as response:
                        if response.status == 200:
                            result = await response.json()
                            if result.get("candidates") and result["candidates"][0].get("content") and \
                               result["candidates"][0]["content"].get("parts"):
                                return result["candidates"][0]["content"]["parts"][0]["text"]
                            else:
                                return "Could not generate suggestion."
                        elif response.status == 429: 
                            delay = INITIAL_RETRY_DELAY * (2 ** attempt) 

                            try:
                                error_details = await response.json()
                                for detail in error_details.get("error", {}).get("details", []):
                                    if "@type" in detail and "RetryInfo" in detail["@type"] and "retryDelay" in detail:
                                        delay_str = detail["retryDelay"]
                                        if delay_str.endswith("s"):
                                            parsed_delay = int(delay_str[:-1])
                                            delay = max(delay, parsed_delay)
                                        break
                            except json.JSONDecodeError:
                                pass
                            
                            await asyncio.sleep(delay)
                            continue 
                        else:
                            error_message = await response.text()
                            if "PERMISSION_DENIED" not in error_message:

                                st.error(f"Error calling LLM API (Status: {response.status}): {error_message}")
                            return "Error generating suggestion."
            except aiohttp.ClientError as e:

                await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
                continue
            except Exception as e:
                st.error(f"An unexpected error occurred for '{missing_keyword}': {e}")
                return "Error generating suggestion."
        
        return "Failed to generate suggestion after multiple retries."


st.set_page_config(
    page_title="Resume Matcher",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap'); /* Added 800 for extra bold */

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        padding: 20px;
    }

    .css-1d391kg { /* Streamlit container for the main content */
        background-color: #ffffff; /* White background for the main content area */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1); /* Stronger shadow */
        margin-bottom: 30px;
    }
    
    .st-emotion-cache-1kyxreq { /* Target the Streamlit primary button */
        background-image: linear-gradient(to right, #4CAF50 0%, #689F38 100%); /* Green gradient */
        color: white;
        padding: 12px 28px; /* Slightly larger padding */
        border-radius: 10px; /* More rounded */
        border: none;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25); /* Stronger shadow */
        transition: all 0.3s ease;
        font-weight: 700; /* Bold */
        font-size: 1.1em; /* Slightly larger font */
        cursor: pointer;
        letter-spacing: 0.5px;
    }
    .st-emotion-cache-1kyxreq:hover {
        background-image: linear-gradient(to right, #45a049 0%, #5d8e32 100%); /* Darker gradient on hover */
        transform: translateY(-3px); /* More pronounced lift */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3); /* Even stronger shadow */
    }

    h1 {
        color: #1a202c;
        text-align: center;
        font-size: 3em; /* Larger title */
        margin-bottom: 10px;
        font-weight: 800; /* Extra bold */
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.2em; /* Slightly larger subtitle */
        margin-bottom: 30px;
        font-weight: 400;
    }

    h2 {
        color: #2d3748;
        font-size: 2em; /* Larger section headers */
        margin-top: 40px;
        margin-bottom: 20px;
        border-bottom: 2px solid #e2e8f0; /* Lighter border */
        padding-bottom: 8px;
        font-weight: 700;
    }

    h3 {
        color: #4a5568;
        font-size: 1.6em; /* Larger subheaders */
        margin-top: 30px;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    h4 {
        color: #2b6cb0; /* A shade of blue for keyword categories */
        font-size: 1.2em; /* Slightly larger for keyword types */
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: 600;
    }

    /* Streamlit components styling */
    .stFileUploader > div > button { /* Upload button style */
        background-color: #cbd5e0;
        color: #2d3748;
        border-radius: 8px;
        border: 1px solid #a0aec0;
        padding: 8px 15px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stFileUploader > div > button:hover {
        background-color: #e2e8f0;
    }
    
    .stTextArea textarea { /* Text area styling */
        border-radius: 8px;
        border: 1px solid #cbd5e0;
        padding: 10px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .st-emotion-cache-1c9v61p { /* st.info box */
        border-radius: 12px; /* More rounded */
        background-color: #e6f7ff;
        color: #0056b3;
        border-left: 6px solid #007bff; /* Thicker border */
        padding: 15px;
        font-size: 0.95em;
    }
    
    .st-emotion-cache-13hv8yv { /* st.success box */
        border-radius: 12px;
        background-color: #e6ffe6;
        color: #28a745;
        border-left: 6px solid #28a745;
        padding: 15px;
        font-size: 0.95em;
    }
    
    .st-emotion-cache-1jmvejs { /* st.warning box */
        border-radius: 12px;
        background-color: #fff3e6;
        color: #ff8c00;
        border-left: 6px solid #ff8c00;
        padding: 15px;
        font-size: 0.95em;
    }

    .element-container {
        padding-bottom: 15px; /* More space between elements */
    }

    .match-percentage-text {
        font-size: 1.8em; /* Larger for impact */
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 15px;
        text-align: center;
        padding: 10px 0;
        border-bottom: 1px dashed #e2e8f0; /* Subtle separator */
    }
    
    .overall-assessment {
        font-size: 1.4em; /* Larger text */
        font-weight: 700;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
        padding: 12px;
        border-radius: 12px; /* More rounded */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Added shadow */
    }
    .overall-assessment.appropriate {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .overall-assessment.not-appropriate {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }

    /* Custom styling for expanders */
    .streamlit-expanderHeader {
        background-color: #e2e8f0;
        border-radius: 10px; /* More rounded */
        padding: 12px;
        margin-top: 20px;
        font-weight: 700;
        color: #2d3748;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); /* Subtle shadow */
    }

    /* Styling for the spinner */
    .stSpinner > div > div {
        color: #4CAF50; /* Spinner color */
        font-size: 1.1em;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.85em;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Resume Matcher</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Effortlessly find the best candidates by comparing resumes with job descriptions using advanced AI.</p>", unsafe_allow_html=True)


st.header("Job Description")
col_jd_upload, col_jd_text = st.columns(2) 

with col_jd_upload:
    st.markdown("<p style='font-weight: 600; margin-bottom: 5px;'>Upload PDF</p>", unsafe_allow_html=True)
    job_desc_upload = st.file_uploader(" ", type=["pdf"], key="jd_pdf_upload", label_visibility="collapsed")
with col_jd_text:
    st.markdown("<p style='font-weight: 600; margin-bottom: 5px;'>Or Paste Text</p>", unsafe_allow_html=True)
    job_desc_text_input = st.text_area(" ", height=300, key="jd_text_area", placeholder="Paste your job description here...", label_visibility="collapsed")

job_description_content = ""
if job_desc_upload is not None:
    job_description_content = extract_text_from_pdf(io.BytesIO(job_desc_upload.read()))
elif job_desc_text_input:
    job_description_content = job_desc_text_input

if not job_description_content:
    st.warning("Please upload or paste a job description to begin the analysis.")

st.header("Resumes")
st.markdown("<p style='font-weight: 600; margin-bottom: 5px;'>Upload one or more resumes in PDF format.</p>", unsafe_allow_html=True)
resume_uploads = st.file_uploader(" ", type=["pdf"], accept_multiple_files=True, key="resume_pdf_upload", label_visibility="collapsed")

if not resume_uploads and not job_description_content:
    st.info("Upload a job description and at least one resume to get started.")

st.markdown("---") 
if st.button("Calculate Match & Get Suggestions!", use_container_width=True, help="Click to analyze the uploaded documents and get feedback."):
    if job_description_content and resume_uploads:
        st.subheader("Analysis Results:")
        results_df_data = []

        with st.spinner("Analyzing resumes and generating AI suggestions... This might take a moment depending on the number of resumes and complexity."):
            for i, resume_file in enumerate(resume_uploads):
                st.markdown(f"---") 
                st.markdown(f"### Resume {i + 1}: <span style='color: #4a5568;'>{resume_file.name}</span>", unsafe_allow_html=True)
                resume_content = extract_text_from_pdf(io.BytesIO(resume_file.read()))

                if resume_content:
                    match_percent, matching_kw, missing_kw, extra_kw = \
                        calculate_match_and_keywords(job_description_content, resume_content, nlp)

                    st.markdown(f"<p class='match-percentage-text'>Match Percentage: <span style='color:#4CAF50;'>{match_percent:.2f}%</span></p>", unsafe_allow_html=True)

                    if match_percent >= 60: 
                        st.markdown("<p class='overall-assessment appropriate'>Overall Assessment: Appropriate</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p class='overall-assessment not-appropriate'>Overall Assessment: Not Appropriate</p>", unsafe_allow_html=True)


                    if match_percent >= 75:
                        st.progress(match_percent / 100, text="**Excellent Match!** This resume shows a strong alignment.")
                        st.success("This candidate's profile is highly suitable for the role.")
                    elif match_percent >= 50:
                        st.progress(match_percent / 100, text="**Good Match!** Solid alignment, some areas for improvement.")
                        st.info("A good match, with potential to strengthen specific areas.")
                    else:
                        st.progress(match_percent / 100, text="**Needs Improvement.** Consider tailoring more.")
                        st.warning("This resume has a lower match score. Significant tailoring is recommended.")

                    with st.expander("View Keyword Details"):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("#### Matching Keywords")
                            if matching_kw:
                                for kw in sorted(matching_kw):
                                    st.markdown(f"- **{kw}**")
                            else:
                                st.write("No direct matching keywords found.")

                        with col2:
                            st.markdown("#### Missing Keywords")
                            if missing_kw:
                                for kw in sorted(missing_kw):
                                    st.markdown(f"- `{kw}`") 
                            else:
                                st.write("All job description keywords found!")

                        with col3:
                            st.markdown("#### Extra Keywords (in Resume)")
                            if extra_kw:
                                for kw in sorted(extra_kw):
                                    st.markdown(f"- *{kw}*")
                            else:
                                st.write("No significantly extra keywords in resume.")

                    st.markdown("---")
                    st.markdown("#### Suggestions for Improvement ")
                    if missing_kw:
                        st.info("Here are some suggestions to enhance your resume by adding context for missing keywords.")
                        
                        suggestions_tasks = [
                            get_resume_enhancement_suggestion(kw, job_description_content)
                            for kw in sorted(missing_kw)
                        ]
                        
                        suggestions = asyncio.run(asyncio.gather(*suggestions_tasks))

                        for kw, suggestion in zip(sorted(missing_kw), suggestions):
                            st.markdown(f"**{kw.capitalize()}:** {suggestion}")
                    else:
                        st.success("Great job! Your resume covers all key aspects of the job description.")

                    results_df_data.append({
                        "Resume Name": resume_file.name,
                        "Match Percentage": match_percent,
                        "Appropriate": "Appropriate" if match_percent >= 60 else "Not Appropriate", 
                        "Matching Keywords Count": len(matching_kw),
                        "Missing Keywords Count": len(missing_kw),
                        "Extra Keywords Count": len(extra_kw),
                        "Matching Keywords": ", ".join(sorted(matching_kw)),
                        "Missing Keywords": ", ".join(sorted(missing_kw)),
                        "Extra Keywords": ", ".join(sorted(extra_kw))
                    })
                else:
                    st.error(f"Could not process Resume {i + 1}: {resume_file.name}. Please ensure it's a valid PDF.")
                st.markdown("---")

        if results_df_data:
            st.subheader("Summary of All Resume Matches")
            results_df = pd.DataFrame(results_df_data)
            st.dataframe(results_df, use_container_width=True) 

            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download All Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="resume_match_results.csv",
                mime="text/csv",
                help="Click to download a CSV file containing all analysis results."
            )
    elif not job_description_content:
        st.error("Please provide the Job Description first.")
    elif not resume_uploads:
        st.error("Please upload at least one Resume.")

st.markdown("---")
