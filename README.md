# ScholarMatch - AI/ML College Scholarship Recommendation Platform

A production-ready, full-stack AI/ML-powered college scholarship recommendation website designed for undergraduate students. Inspired by *MyScheme*, ScholarMatch couples **hard rule-based eligibility verification** with **scikit-learn / XGBoost machine learning recommendation models** to guide students directly to verified scholarships they qualify for.

LINK TO LOVE PROJECT :https://scholarshipapp-uauv.onrender.com

---

## 🌟 Key Features

1. **Hybrid AI & Rule-Based Recommendation Engine**:
   - **Hard Eligibility Engine**: Evaluates strict statutory constraints (gender exclusivity, branch, year of study, CGPA, qualifying percentage, income ceiling, domicile, hostel residency, PwD disability percentage, BPL status). Rule verification strictly takes precedence over ML scores to prevent ineligible recommendations.
   - **ML Recommendation Pipeline**: Comparative evaluation of `Logistic Regression`, `RandomForest`, and `XGBoost` classifiers for all 32 scholarship targets, with per-scholarship best model selection via 5-fold stratified CV.
   - **Transparent Rule-Based Scoring**: High-precision deterministic scoring for 25+ additional scholarships not present in the training targets.

2. **5-Step Interactive Questionnaire Wizard**:
   - Step 1: Basic Information (Gender, Age, Domicile)
   - Step 2: Academic Information (Branch, Year, CGPA, 12th/Diploma %)
   - Step 3: Financial Background (Annual Family Income, BPL Status)
   - Step 4: Category & Inclusion (General/OBC/SC/ST, PwD Certification & %)
   - Step 5: College Residency & Live Summary Review before Submission
   - Pure Vanilla JavaScript validation, step progress bar, and dynamic conditional fields.

3. **Intelligent Results Dashboard**:
   - Summary statistics (Matching Count, AI-Scored Picks, Potential Financial Aid sum).
   - Filter tabs: *All Eligible Matches*, *AI Top Recommendations*, *Rule-Based Matches*, *Ineligible Opportunities*.
   - Scholarship cards detailing *"Why you're eligible"* checklists and required documents.

4. **Detailed Scholarship Information & Verified Official Links**:
   - Breakdown of benefits, full eligibility matrix, document checklists, and application timelines.
   - Verified 1-click **"Apply on Official Portal"** links opening directly in a new secure tab.
   - Missing/unverified links display `"Official application link currently unavailable"` gracefully without creating fake or broken links.

5. **Searchable & Filterable Scholarship Directory**:
   - Live client-side keyword search across name, provider, and description.
   - Instant filtering by Engineering Branch, Gender, and Social Category.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.9+ with **Flask**
- **Machine Learning**: **scikit-learn** (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `MultiOutputClassifier`), **XGBoost** (`XGBClassifier`), **pandas**, **numpy**, **joblib**
- **Templating**: **Jinja2** with custom filters (e.g. Indian Rupee formatting `₹85,545`)
- **Frontend / Styling**: Semantic **HTML5**, **Custom Responsive CSS** (No external CSS frameworks like Tailwind or Bootstrap)
- **Client-Side Scripting**: **Vanilla JavaScript** (No frontend frameworks like React or Node.js)
- **Testing**: Python standard `unittest` suite (21 unit and integration tests)

---

## 📂 Project Structure

```
beautiful-hubble/
│
├── app.py                     # Main Flask application & route controllers
├── requirements.txt           # Python dependencies
├── README.md                  # Detailed project documentation
├── .env.example               # Environment variables template
│
├── ml/
│   ├── scholarship_system.pkl  # Trained multi-model ML artifact (compressed)
│   ├── model_loader.py         # Singleton model loader (loaded once at startup)
│   ├── predict.py              # Inference: feature engineering → preprocess → predict
│   ├── features.py             # Shared constants & single authoritative engineer_features()
│   ├── pipeline.py             # Comparative training pipeline (LR, RF, XGBoost)
│   ├── model_comparison.csv    # Full CV + test results comparison table
│   └── feature_importance.csv  # Top feature importances per scholarship
│
├── data/
│   ├── scholarship_dataset.csv # 5,000 students × 45 columns (13 raw + 32 targets)
│   └── scholarships.json       # 32+ comprehensive scholarship database
│
├── eligibility/
│   ├── __init__.py
│   └── eligibility_engine.py   # Hard eligibility constraint rules & ML ranking
│
├── templates/
│   ├── base.html               # Base layout with PCCOE branding, navbar & footer
│   ├── index.html              # Landing page with hero & 4-step workflow
│   ├── questionnaire.html      # 5-step multi-step questionnaire wizard
│   ├── results.html            # Recommendation matches & filter dashboard
│   ├── scholarships.html       # Searchable & filterable scholarship directory
│   ├── scholarship_detail.html # Comprehensive scholarship view & official apply button
│   ├── about.html              # Methodology & research limitations
│   ├── 404.html                # Custom 404 Not Found page
│   └── 500.html                # Custom 500 Server Error page
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom responsive CSS design system
│   ├── images/
│   │   └── pccoe_logo.png      # PCCOE institutional crest logo
│   └── js/
│       ├── questionnaire.js    # Multi-step wizard navigation & live validation
│       └── main.js             # Directory search, filter tabs & interaction
│
└── tests/
    ├── test_eligibility.py     # Unit tests for statutory eligibility rules
    ├── test_prediction.py      # Unit tests for ML feature engineering & inference
    └── test_routes.py          # End-to-end route & integration test suite
```

---

## 🤖 ML Architecture & Comparative Pipeline

- **Approach**: Comparative evaluation of **Logistic Regression**, **Random Forest**, and **XGBoost** per scholarship target.
- **Validation**: 5-fold Stratified Cross-Validation with F1 as primary selection metric.
- **Feature Engineering**: 16 derived features computed dynamically from 13 raw attributes (never stored in CSV).
- **PCA Experiments**: Each model tested with and without PCA (95% variance retention).
- **Threshold Optimization**: Per-scholarship thresholds tuned on CV predictions only, frozen before test evaluation.
- **Preprocessing**: StandardScaler + OneHotEncoder, fitted inside each CV fold (no data leakage).
- **Model Selection**: Best model per scholarship selected by mean CV F1 score.
- **Final Evaluation**: Selected models retrained on full 80% training set, evaluated once on held-out 20% test set.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9 or higher installed.

### 2. Create and Activate Virtual Environment

**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
```powershell
copy .env.example .env
```

### 5. Retrain Models (Optional)
To retrain the comparative ML pipeline:
```bash
python ml/pipeline.py
```

### 6. Run the Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 7. Start the Flask Server
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
/ (Landing Page)
       │
       ▼ [Find My Scholarships]
/questionnaire (5-Step Interactive Form)
       │
       ▼ [Submit Profile]
POST /recommend (Flask Backend)
       ├── 1. Validate inputs & coerce data types
       ├── 2. Evaluate Hard Eligibility Engine
       ├── 3. Generate ML Recommendation Scores (predict_proba)
       ├── 4. Apply Hybrid Ranking & Store Profile in Session
       │
       ▼ [Redirect]
/results (Your Scholarship Matches)
       │
       ▼ [View Scholarship]
/scholarships/<scholarship_id> (Full Details, Checklist & Deadline)
       │
       ▼ [Apply / Visit Official Portal]
Official Scholarship Application Portal (External Website in new tab)
```

---

## 🔬 Research & Data Limitations

> [!NOTE]
> The initial recommendation model uses the supplied synthetic student dataset and its existing scholarship target labels. These labels represent scholarship-match classifications in the dataset and should not be interpreted as actual probabilities of receiving a scholarship.
>
> Official eligibility constraints are handled independently through the rule-based eligibility engine. Future iterations can incorporate real-world application tracking, institutional quotas, and award outcomes to continually train and refine the recommendation model.

---

## 🛡️ Security & Reliability Best Practices

- **Zero Client-Side Framework Overhead**: Fast, clean, native HTML/CSS/JS without unnecessary dependencies.
- **Lightweight Session Storage**: Stores only serialized student profile primitives in cookies (<300 bytes), avoiding 4KB cookie truncation bugs while keeping URLs clean of sensitive personal information.
- **Safe External Navigation**: All outbound links use `target="_blank" rel="noopener noreferrer"`.
- **Model Crash Protection**: If the ML model throws an error, the backend gracefully falls back to the deterministic eligibility rule engine without failing the user's request.
