import os

import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
import docx



# Text Extraction Functions

def extract_text_from_pdf(file):
    """Extract text from a PDF file-like object."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_text_from_docx(file):
    """Extract text from a DOCX file-like object."""
    document = docx.Document(file)
    full_text = []
    for para in document.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


def extract_text(file):
    """
    Generic extractor that checks extension and calls appropriate function.
    Supports .pdf and .docx.
    """
    filename = file.name.lower()
    _, ext = os.path.splitext(filename)

    if ext == ".pdf":
        return extract_text_from_pdf(file)
    elif ext == ".docx":
        return extract_text_from_docx(file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# -----------------------------
# Ranking Functions
# -----------------------------

# Legacy TF-IDF-based ranking (kept for comparison / toggle)
def rank_resumes_tfidf(job_description, resumes_text):
    """
    Rank resumes using TF-IDF + cosine similarity.
    Returns a list of similarity scores aligned with resumes_text.
    """
    documents = [job_description] + resumes_text
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    cosine_similarities = cosine_similarity(
        [job_description_vector],
        resume_vectors
    ).flatten()  # shape: (num_resumes,)
    return cosine_similarities


# Load Sentence Transformer model once (cached)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


model = load_embedding_model()


def rank_resumes_embeddings(job_description, resumes_text):
    """
    Rank resumes using Sentence Transformer embeddings + cosine similarity.
    Returns a sorted list of (index, score) pairs.
    """
    jd_emb = model.encode(
        job_description,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    resumes_emb = model.encode(
        resumes_text,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    cosine_scores = util.cos_sim(jd_emb, resumes_emb)[0]  # shape: (num_resumes,)
    scored = [(i, float(score)) for i, score in enumerate(cosine_scores)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# -----------------------------
# Simple Skill Extraction (optional nice add-on)
# -----------------------------

SKILLS = [
    "python", "java", "c++", "c", "javascript", "html", "css",
    "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "data science",
    "pandas", "numpy", "scikit-learn", "django", "flask",
    "react", "node.js", "excel", "power bi", "tableau"
]


def extract_skills(text):
    text_low = text.lower()
    found = []
    for skill in SKILLS:
        if skill in text_low:
            found.append(skill)
    return sorted(set(found))


# -----------------------------
# Streamlit App
# -----------------------------

st.title("AI Resume Screening and Candidate Ranking System")

st.markdown(
    """
This app helps you **screen and rank resumes** based on a job description.

- Upload multiple **PDF or DOCX** resumes  
- Enter the **job description**  
- Choose a **ranking method**:
  - 🧠 *Embeddings (Sentence Transformers)* – Recommended for semantic matching  
  - ✍️ *TF-IDF* – Simple keyword-based baseline
"""
)

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description", height=200)

# Ranking method selection
st.header("Ranking Method")
method = st.selectbox(
    "Select the method to rank candidates",
    ["Embeddings (recommended)", "TF-IDF"]
)

# File uploader
# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload PDF/DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        height: 3em;
        width: 10em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# New button
rank_button = st.button("Rank Resumes")

st.markdown("---")


# Ranking logic (triggered by button)
if rank_button:
    if not job_description:
        st.error("Please enter a job description before ranking.")
    elif not uploaded_files:
        st.error("Please upload at least one resume before ranking.")
    else:
        st.header("Ranking Results")
        try:
            resumes_text = []
            resume_names = []

            # Extract text from each uploaded resume
            for file in uploaded_files:
                try:
                    text = extract_text(file)
                    if text and text.strip():
                        resumes_text.append(text)
                        resume_names.append(file.name)
                    else:
                        st.warning(f"{file.name} appears to be empty or unreadable.")
                except Exception as e:
                    st.error(f"Error reading {file.name}: {str(e)}")

            if not resumes_text:
                st.error("No valid resumes could be read.")
            else:
                if method == "Embeddings (recommended)":
                    st.caption("Ranking using Sentence-Transformer embeddings (semantic similarity).")
                    scored = rank_resumes_embeddings(job_description, resumes_text)

                    for rank, (idx, score) in enumerate(scored, start=1):
                        name = resume_names[idx]
                        skills = extract_skills(resumes_text[idx])

                        st.markdown(f"### {rank}. {name}")
                        st.write(f"**Match Score:** {score:.4f}")
                        if skills:
                            st.caption(f"Detected skills: {', '.join(skills)}")
                        st.markdown("---")

                else:  # TF-IDF
                    st.caption("Ranking using TF-IDF + cosine similarity (keyword-based).")
                    cosine_similarities = rank_resumes_tfidf(job_description, resumes_text)

                    indexed_scores = list(enumerate(cosine_similarities))
                    indexed_scores.sort(key=lambda x: x[1], reverse=True)

                    for rank, (idx, score) in enumerate(indexed_scores, start=1):
                        name = resume_names[idx]
                        skills = extract_skills(resumes_text[idx])

                        st.markdown(f"### {rank}. {name}")
                        st.write(f"**Match Score:** {score:.4f}")
                        if skills:
                            st.caption(f"Detected skills: {', '.join(skills)}")
                        st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
