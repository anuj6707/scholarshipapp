# ScholarMatch - AI/ML College Scholarship Recommendation Platform

A production-ready, full-stack AI/ML-powered college scholarship recommendation website designed for undergraduate engineering and STEM students. ScholarMatch pairs **hard rule-based statutory eligibility verification** with **calibrated multi-target machine learning classifiers** and **SHAP (SHapley Additive exPlanations) explainability** to guide students to scholarships they qualify for.

**Live Project URL:** [https://scholarshipapp-uauv.onrender.com](https://scholarshipapp-uauv.onrender.com)

---

## 🌟 Key Features

### 1. Dual Recommendation Architecture
ScholarMatch supports two distinct evaluation modes:

* **Mode A: Statutory Rules + ML Hybrid Engine (Default — `/results` or `/results?mode=rules`)**
  * Evaluates hard statutory constraints first (gender restrictions, engineering branch, academic year of study, CGPA, qualifying percentage, annual family income ceiling, state domicile, hostel residency, PwD disability percentage, and BPL status).
  * Strict rule verification prevents ineligible recommendations. Only applications meeting 100% of legal constraints proceed to ML scoring and ranking.
* **Mode B: Autonomous ML Confidence Threshold Engine with SHAP (`/results?mode=confidence`)**
  * Discards hardcoded rule logic and lets calibrated machine learning models evaluate eligibility autonomously.
  * Employs per-scholarship decision thresholds ($\tau^* \in [0.30, 0.70]$) with an operational safety floor ($\tau_{\text{eff}} \ge 0.40$).
  * Generates plain-English qualifying and unmet reasons using **SHAP (SHapley Additive exPlanations)** feature attributions.

### 2. Explainable AI (XAI) with SHAP TreeExplainer
* Uses `shap.TreeExplainer` on fitted tree ensembles (XGBoost and Random Forest) to compute exact feature attribution values per student profile.
* **Why You Qualify (Positive SHAP):** Highlights top features that elevated the student's score above the threshold (e.g., family income need, high CGPA, prioritized branch).
* **Unmet Requirements (Negative SHAP):** Identifies exact limiting factors that pulled confidence below the threshold (e.g., year of study mismatch, income exceeding threshold, demographic criteria).
* In-memory caching delivers full SHAP explanations across all 32 targets in **~20 milliseconds** without latency.

### 3. 5-Step Interactive Questionnaire Wizard
* **Step 1:** Demographics (Gender, Age, Domicile State)
* **Step 2:** Academic Information (Branch, Year of Study, CGPA, 12th/Diploma %)
* **Step 3:** Financial Background (Annual Family Income, BPL Certification)
* **Step 4:** Inclusion & Quotas (Social Category, PwD Certification & Disability %)
* **Step 5:** Residential Context & Live Pre-Flight Review before submission
* Built with pure Vanilla JavaScript validation, step progress indicators, and dynamic conditional fields.

### 4. Intelligent Results Dashboard
* Financial aid statistics: Matched Scholarship Count, High-Confidence Picks ($\ge 80\%$), and Total Potential Benefit sum (formatted in Indian Rupees e.g., `₹2,45,000`).
* Segmented filter tabs: *Top Matches (80%+)*, *All Eligible Matches*, and *Ineligible Opportunities*.
* Card breakdowns displaying match percentage badges (`⚡ Match Score: XX%`), requirement checklists, and required documentation.

### 5. Searchable & Filterable Scholarship Directory
* Live client-side keyword search across scholarship names, providers, and descriptions.
* Instant multi-tag filtering by Engineering Branch and Social Category.
* Detailed scheme modal views with verified 1-click **"Apply on Official Portal"** links opening safely in a new tab.

---

## 🛠️ Technology Stack

* **Backend Framework:** Python 3.9+ with **Flask**
* **Machine Learning & Explainability:** 
  * **scikit-learn** (`StandardScaler`, `OneHotEncoder`, `StratifiedKFold`, metrics)
  * **XGBoost** (`XGBClassifier`)
  * **SHAP** (`TreeExplainer` for local feature attribution)
  * **pandas**, **numpy**, **joblib**
* **Templating:** **Jinja2** with custom filters (e.g., Indian Rupee numbering format `₹1,50,000`)
* **Frontend Design System:** Semantic **HTML5**, **Custom Responsive CSS3** (No heavy frameworks like Bootstrap or Tailwind)
* **Client-Side Scripting:** **Vanilla JavaScript** (Zero external runtime dependencies)
* **Automated Testing:** Python standard `unittest` suite (21 unit and integration tests)

---

## 📂 Project Structure

```
beautiful-hubble/
├── app.py                      # Primary Flask router, session manager & pipeline controller
├── requirements.txt            # Python dependencies (Flask, scikit-learn, xgboost, shap, etc.)
├── README.md                   # Comprehensive project documentation
├── .env.example                # Environment variables template
│
├── data/
│   ├── scholarship_dataset.csv # 5,512 students × 45 columns (13 raw attributes + 32 targets)
│   └── scholarships.json       # Formal statutory constraints database for all 32 schemes
│
├── ml/
│   ├── features.py             # Single authoritative feature engineering (13 -> 29 features)
│   ├── pipeline.py             # Multithreaded comparative training & threshold tuning pipeline
│   ├── predict.py              # Inference engine (SHAP explainability + thresholding)
│   ├── model_loader.py         # Singleton model cache and target lookup mapping
│   ├── scholarship_system.pkl  # Production serialized artifact (~1.39 MB)
│   ├── model_comparison.csv    # Cross-validation benchmark logs across all model configs
│   └── feature_importance.csv  # Global Gini feature importance rankings
│
├── eligibility/
│   ├── __init__.py
│   └── eligibility_engine.py   # Statutory rule engine, hybrid blending & tie-breaking
│
├── templates/
│   ├── base.html               # Master layout with PCCOE institutional branding
│   ├── index.html              # Landing page with hero section & 4-step workflow guide
│   ├── questionnaire.html      # 5-step interactive wizard questionnaire
│   ├── results.html            # Results dashboard with cards, metrics & SHAP checklists
│   ├── scholarships.html       # Searchable directory with branch & category filters
│   ├── scholarship_detail.html # Comprehensive scheme breakdown and official apply portal
│   ├── about.html              # Research methodology & academic background
│   ├── 404.html                # Custom 404 Not Found error page
│   └── 500.html                # Custom 500 Server Error page
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom responsive CSS design system
│   ├── images/
│   │   └── pccoe_logo.png      # PCCOE institutional crest logo
│   └── js/
│       ├── questionnaire.js    # Multi-step wizard navigation, validation & live review
│       └── main.js             # Directory live search, filter tabs & interaction
│
└── tests/
    ├── test_eligibility.py     # 6 unit tests for hard statutory rule enforcement
    ├── test_prediction.py      # 4 unit tests for feature transforms, inference & SHAP
    └── test_routes.py          # 11 integration tests for web endpoints & session states
```

---

## 🤖 Machine Learning Architecture & Benchmarks

### 1. Dataset & Feature Engineering
* **Dataset Size:** 5,512 instances (5,000 synthetic stratified profiles + 512 verified empirical applicant records from national programs like *AICTE Pragati* and *Glow & Lovely*).
* **Balanced Label Noise:** 2.8% balanced boundary noise injected across target columns to model real-world clerical variance and prevent artificial 100% classification certainty.
* **Zero Leakage:** No engineered features are stored in the CSV. Exactly 16 derived features (`academic_score`, `financial_hardship_score`, `merit_percentile`, `need_and_merit`, `stem_branch`, etc.) are computed on-the-fly via `ml/features.py`.

### 2. Multi-Target Model Ensemble
* Evaluated **Logistic Regression**, **Random Forest**, and **XGBoost** across 5-fold stratified cross-validation.
* **XGBoost** was selected for **31 of 32 scholarships**; **Random Forest** for **1 scholarship** (`loreal_women_science`).
* Models are serialized into an explicit preprocessor dictionary (avoiding scikit-learn cross-version unpickling errors on cloud hosts like Render).

### 3. Comprehensive Performance Metrics (Held-Out 20% Test Cohort, $N=1,103$)

| Metric | Macro Average | Min Value | Max Value |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **98.47%** | 90.57% | 99.91% |
| **Precision** | **94.12%** | 85.71% | 100.00% |
| **Recall (Sensitivity)** | **95.08%** | 78.57% | 100.00% |
| **F1-Score** | **94.53%** | 84.62% | 97.67% |
| **ROC-AUC** | **0.9855** | 0.9442 | 1.0000 |
| **Optimal Threshold ($\tau^*$)** | **0.481** | 0.300 | 0.700 |

* **Model Artifact Size:** Compact **1.39 MB** (`ml/scholarship_system.pkl`).
* **Training Time:** **~6.7 minutes** using multithreaded parallel execution (`ml/pipeline.py`).
* **Single-Sample Inference Time:** **~20 milliseconds** (including SHAP calculation).

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.9 or higher is installed on your system.

### 2. Clone Repository & Setup Environment

**Windows (PowerShell):**
```powershell
git clone https://github.com/anuj6707/scholarshipapp.git
cd scholarshipapp
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
git clone https://github.com/anuj6707/scholarshipapp.git
cd scholarshipapp
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Retrain Models in Parallel (Optional)
To run the full 5-fold cross-validation and threshold optimization pipeline:
```bash
python ml/pipeline.py
```

### 5. Run the Automated Test Suite
To execute all 21 unit and integration tests:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 6. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧭 Complete User Flow

```
/ (Landing Page & Workflow Guide)
       │
       ▼ [Find My Scholarships]
/questionnaire (5-Step Multi-Step Wizard)
       │
       ▼ [Submit Profile]
POST /recommend (Backend Validation & Coercion)
       ├── 1. Coerce & validate inputs (Bounds & Categories)
       ├── 2. Store lightweight student profile in session (<300 bytes)
       │
       ▼ [Redirect]
/results (Recommendation Dashboard)
       ├── Mode A (Default): Statutory Rules Engine + ML Ranking
       └── Mode B (?mode=confidence): Autonomous Thresholds + SHAP Explanations
       │
       ▼ [View Scholarship Details]
/scholarships/<scholarship_id> (Checklists, Deadlines & Verification Matrix)
       │
       ▼ [Apply on Official Portal]
External Official Scholarship Portal (Opens safely in new tab)
```

---

## 🛡️ Reliability & Security Best Practices

* **Zero Client-Side Framework Bloat:** Fast initial page loads using semantic HTML5 and vanilla JavaScript without React or Angular runtime overhead.
* **Lightweight Session Cookies:** Stores only primitive student attributes in signed cookies (<300 bytes), avoiding the 4KB HTTP cookie overflow issue while keeping personal data off the URL query string.
* **Safe External Navigation:** Outbound links use `target="_blank" rel="noopener noreferrer"` to protect browser security context.
* **Model Crash Protection:** If an ML model raises an exception, the system gracefully falls back to deterministic rule evaluations without failing user requests.
