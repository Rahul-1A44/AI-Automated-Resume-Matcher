AI-Powered Resume Matcher 


This Streamlit application helps recruiters and job seekers efficiently compare resumes against job descriptions. It leverages Natural Language Processing (NLP) to calculate a match percentage, identify key matching and missing keywords, and even provides AI-generated suggestions to enhance resumes for specific roles.

 Features
PDF Text Extraction: Upload job descriptions and multiple resumes in PDF format.

Intelligent Matching: Calculates a cosine similarity score between the job description and each resume using TF-IDF vectorization.

Keyword Analysis:

Identifies matching keywords present in both the job description and the resume.

Highlights missing keywords from the job description that are absent in the resume.

Lists extra keywords found in the resume but not explicitly in the job description.

AI-Powered Suggestions: Utilizes the Gemini API to provide concise, actionable suggestions for improving resumes by incorporating missing keywords with relevant context.

Interactive UI: Built with Streamlit for an intuitive and user-friendly experience.

Summary Table & CSV Export: Presents an overview of all resume matches in a sortable table and allows downloading the full results as a CSV file for easy analysis.

🛠️ Technologies Used
Python 3.x

Streamlit: For creating the interactive web application.

spaCy: For advanced NLP tasks like tokenization, lemmatization, noun chunking, and named entity recognition.

scikit-learn: For TfidfVectorizer and cosine_similarity to calculate document similarity.

PyPDF2: For extracting text from PDF documents.

aiohttp & asyncio: For asynchronous handling of multiple API calls to the Gemini LLM.

nest_asyncio: To allow nested event loops in Streamlit's environment.

pandas: For data handling and result summary tables.

Google Gemini API (gemini-2.0-flash): For AI-driven resume enhancement suggestions.

🚀 Installation & Setup
Follow these steps to set up and run the application locally.

1. Clone the Repository
First, clone this GitHub repository to your local machine:

git clone https://github.com/Rahul-1A44/AI-Automated-Resume-Matcher.git
cd AI-Automated-Resume-Matcher

2. Create a Virtual Environment 
It's best practice to create a virtual environment to manage dependencies:

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS / Linux:

source venv/bin/activate

3. Install Required Libraries
Install all the necessary Python packages using pip:

pip install -r requirements.txt

(If you don't have a requirements.txt file, you can create one by running pip freeze > requirements.txt after installing them manually, or just install them directly:)

pip install streamlit PyPDF2 spacy pandas scikit-learn aiohttp nest_asyncio

4. Download the SpaCy NLP Model
The application uses the en_core_web_sm spaCy model. Download it using the following command:

python -m spacy download en_core_web_sm

5. Set up Gemini API Key
The application uses the Google Gemini API for AI suggestions. You need to obtain an API key and configure it securely with Streamlit:

Get an API Key: Visit the Google AI Studio or Google Cloud Console to generate a Gemini API key.

Create secrets.toml: In your project's root directory (the AI-Automated-Resume-Matcher folder), create a new directory named .streamlit. Inside .streamlit, create a file named secrets.toml.

Add Your API Key: Open secrets.toml and add your Gemini API key in the following format:

# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

Replace YOUR_GEMINI_API_KEY_HERE with your actual key.

Security Note: secrets.toml is meant for local development and Streamlit Cloud. Do NOT commit this file to public GitHub repositories! It's already in the .gitignore for this project to prevent accidental uploads.

▶️ How to Run the Application
Once all installations and setup are complete, run the Streamlit application from your project's root directory:

streamlit run app.py

This command will open the application in your default web browser (usually at http://localhost:8501).

👨‍💻 Usage
Upload Job Description: In the "Job Description" section, you can either:

Upload a PDF file: Click "Upload PDF" and select your job description.

Paste Text: Paste the job description text directly into the provided text area.

Upload Resume(s): In the "Resume(s)" section, click "Upload one or more resumes" and select one or multiple resume PDF files.

Analyze: Click the "Calculate Match & Get Suggestions!" button.

View Results:

The app will process each resume, showing its match percentage, an overall assessment, and detailed keyword breakdowns (matching, missing, extra).

For missing keywords, AI-powered suggestions will be provided to help tailor the resume.

A "Summary of All Resume Matches" table will display an overview of all processed resumes.

Download Results: Use the "📥 Download All Results as CSV" button to export the summary table.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or open a pull request.

📄 License
This project is open-source. 
✉️ Contact & Credits
Author: Rahul-1A44 

Feel free to connect on [LinkedIn](https://www.linkedin.com/in/rahul-gupta-86a14b2b1/) or other platforms.
