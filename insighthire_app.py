import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import docx2txt

# --- Page Config ---
st.set_page_config(page_title="InsightHire — Job Matcher", layout="centered")

# --- Load Model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- File Readers ---
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return " ".join(pages)

def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

def extract_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported format. Please upload a PDF or DOCX file.")
        st.stop()

# --- Soft Skills Matching ---
SOFT_SKILL_PROMPTS = [
    "teamwork", "communication", "leadership", "problem solving",
    "volunteering", "critical thinking", "time management", "collaboration"
]

@st.cache_data
def compute_soft_skill_score(resume_text):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    scores = [
        util.cos_sim(resume_emb, model.encode(skill, convert_to_tensor=True)).item()
        for skill in SOFT_SKILL_PROMPTS
    ]
    return sum(scores) / len(scores)

# --- Semantic Similarity ---
def compute_semantic_scores(resume_text, job_descs):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_embs = model.encode(job_descs.tolist(), convert_to_tensor=True)
    return util.cos_sim(resume_emb, job_embs).squeeze().cpu().numpy()

# --- Main Matching Pipeline ---
def match_resume_to_jobs(resume_text, df_jobs, semantic_weight=0.8):
    soft_weight = 1 - semantic_weight

    semantic_scores = compute_semantic_scores(resume_text, df_jobs["Job Desc"])
    soft_score = compute_soft_skill_score(resume_text)

    df_jobs = df_jobs.copy()
    df_jobs["Semantic Score (%)"] = [round(s * 100, 2) for s in semantic_scores]
    df_jobs["Soft Skill Score (%)"] = round(soft_score * 100, 2)
    df_jobs["Skill Fit Score (%)"] = [
        round((semantic_weight * sem + soft_weight * soft_score) * 100, 2)
        for sem in semantic_scores
    ]

    return df_jobs.sort_values(by="Skill Fit Score (%)", ascending=False)

# --- UI ---
st.title("🎯 InsightHire — Semantic Resume-to-Job Matcher")
st.markdown("Upload your **resume** and a **job dataset CSV** to find your best-fit roles using AI-powered matching.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    uploaded_resume = st.file_uploader("📄 Upload Resume", type=["pdf", "docx"])
with col2:
    uploaded_csv = st.file_uploader("📊 Upload Job Dataset (CSV)", type=["csv"])

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    semantic_weight = st.slider(
        "Semantic vs Soft Skill Weight",
        min_value=0.5, max_value=1.0, value=0.8, step=0.05,
        help="Higher = more weight on job description match. Lower = more weight on soft skills."
    )
    role_filter = st.text_input(
        "Filter job titles (regex)",
        value="analyst|data|bi|engineer|consultant",
        help="Comma-separated keywords to filter relevant roles."
    )
    top_n = st.slider("Number of results to show", min_value=5, max_value=25, value=10)

# --- Run Matching ---
if uploaded_resume and uploaded_csv:
    with st.spinner("Reading resume..."):
        resume_text = extract_resume_text(uploaded_resume)

    if not resume_text.strip():
        st.error("Could not extract text from your resume. Please check the file.")
        st.stop()

    with st.spinner("Loading job dataset..."):
        df = pd.read_csv(uploaded_csv)

    required_cols = {"Job Title", "Job Desc", "Company Name", "Job Location"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_cols}. Found: {set(df.columns)}")
        st.stop()

    df_filtered = df[df["Job Title"].str.contains(role_filter, case=False, na=False)]

    if df_filtered.empty:
        st.warning("No jobs matched your filter. Try broadening the role filter in the sidebar.")
        st.stop()

    with st.spinner(f"Matching your resume against {len(df_filtered)} roles..."):
        result_df = match_resume_to_jobs(resume_text, df_filtered, semantic_weight)

    st.success(f"✅ Matching complete — showing top {top_n} roles")

    display_cols = ["Job Title", "Company Name", "Job Location", "Skill Fit Score (%)", "Semantic Score (%)"]
    st.dataframe(
        result_df[display_cols].head(top_n).reset_index(drop=True),
        use_container_width=True
    )

    # --- Download Results ---
    csv_out = result_df[display_cols].head(top_n).to_csv(index=False)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_out,
        file_name="insighthire_matches.csv",
        mime="text/csv"
    )

    # --- Resume Preview ---
    with st.expander("📃 View extracted resume text"):
        st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

else:
    st.info("👆 Upload both your resume and a job dataset CSV to get started.")
