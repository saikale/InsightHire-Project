# InsightHire — Employability Analytics Platform

> AI-driven platform developed in collaboration with **CeADAR (Centre for Applied Data Analytics Research), University College Cork, Ireland** as part of an MSc in Business Analytics final project.

---

## 🎯 Problem Statement

Graduate employability in Ireland faces a structural gap — candidates struggle to align their skills with job requirements, while employers lack tools to efficiently match talent. InsightHire bridges this gap by combining AI-powered resume analysis, job market intelligence, and predictive analytics into a single platform.

---

## 🏗️ Architecture & Approach

### 1. Data Acquisition
- Scraped live job postings from **LinkedIn and Indeed** using Octoparse
- Parsed candidate resumes using **Google Document AI**

### 2. Data Integration
- Standardised job titles, skills, and eligibility criteria using **Python and manual curation**
- Built a unified schema linking candidate profiles to job market demand

### 3. Modelling & Analysis

| Model | Purpose | Algorithm |
|---|---|---|
| Skill Fit Score | Match candidate skills to job requirements | XGBoost |
| Hiring Trends | Forecast demand by role and sector | ARIMA |
| Role Progression | Predict career pathway likelihood | Random Forest |
| Risk Prediction Index | Flag employability risk factors | Custom scoring |
| Opportunity Index | Surface best-fit opportunities per candidate | Custom scoring |

### 4. Visualisation & Delivery
- Built a **Streamlit web app** for candidate-facing insights
- Integrated **interactive Power BI dashboards** for hiring trend visualisation
- Delivered outputs including skill-fit scores, referral pathways, and upskilling recommendations

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, Random Forest, ARIMA |
| NLP | SpaCy, NLTK |
| Document Parsing | Google Document AI |
| Web Scraping | Octoparse |
| Visualisation | Power BI, Streamlit |

---

## 📊 Key Outputs

- **Skill Fit Score** — quantified alignment between a candidate's profile and a target role
- **Hiring Trend Forecasts** — ARIMA-based demand forecasting by role category
- **Referral Pathway Recommendations** — network-driven job matching insights
- **Upskilling Suggestions** — personalised skill gap recommendations per candidate

---

## 💡 Key Learnings

- Managing ambiguous, real-world datasets across multiple sources
- Agile prototyping and iterative model development
- Stakeholder storytelling with data — translating model outputs into actionable insights
- Ethical handling of candidate and employer data

---

## 🔗 Related

- **Institution:** University College Cork (UCC), Ireland
- **Research Partner:** CeADAR — Centre for Applied Data Analytics Research
- **Programme:** MSc Business Analytics (Grade: 2:1)
