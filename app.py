import os
import secrets
import logging
from flask import Flask, render_template, request, redirect, url_for, session, flash, abort, jsonify
from ml.model_loader import load_system, FEATURE_NAMES
from ml.predict import predict_scholarships
from eligibility.eligibility_engine import (
    get_scholarships,
    get_scholarship_by_id,
    evaluate_and_rank_scholarships,
    check_eligibility
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Pre-load ML recommendation system once on startup
try:
    load_system()
    logger.info("Scholarship ML System pre-loaded successfully during application initialization.")
except Exception as e:
    logger.warning(f"Could not pre-load scholarship system at startup: {e}")

@app.template_filter("currency_inr")
def currency_inr_filter(value):
    """Format integers or numbers to Indian Rupee representation (e.g. ₹1,50,000)."""
    if value is None or value == "":
        return "N/A"
    try:
        val = int(value)
        # Indian numbering formatting
        s = str(val)
        if len(s) <= 3:
            return f"₹{s}"
        last_three = s[-3:]
        remaining = s[:-3]
        out = ""
        while len(remaining) > 2:
            out = "," + remaining[-2:] + out
            remaining = remaining[:-2]
        out = remaining + out + "," + last_three
        return f"₹{out}"
    except (ValueError, TypeError):
        return f"₹{value}"

@app.context_processor
def inject_global_vars():
    """Injects common context variables available in all Jinja templates (No personal name)."""
    return {
        "app_name": "PCCOE Scholarship Portal",
        "college_name": "Pimpri Chinchwad College of Engineering",
        "current_year": 2026
    }

@app.route("/")
def index():
    """Landing page introducing the scholarship recommender."""
    all_scholarships = get_scholarships()
    total_count = len(all_scholarships)
    active_count = sum(1 for s in all_scholarships if s.get("active", True))
    featured = [s for s in all_scholarships if s.get("id") in ["cummins", "siemens", "reliance", "adobe_wit", "lila_poonawala"]]
    return render_template(
        "index.html",
        total_scholarships=total_count,
        active_scholarships=active_count,
        featured_scholarships=featured
    )

@app.route("/questionnaire", methods=["GET"])
def questionnaire():
    """Renders the multi-step questionnaire wizard."""
    student_profile = session.get("student_profile", {})
    return render_template("questionnaire.html", profile=student_profile)

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Processes questionnaire submission:
    1. Validates form data & coerces data types
    2. Constructs student profile
    3. Runs eligibility engine & Machine Learning predictions
    4. Stores lightweight student profile in session
    5. Redirects to /results
    """
    try:
        form = request.form

        # Extract & validate fields
        gender = form.get("gender", "").strip()
        age_str = form.get("age", "").strip()
        domicile = form.get("domicile", "").strip()
        category = form.get("category", "").strip()
        branch = form.get("branch", "").strip()
        year_str = form.get("year", "").strip()
        cgpa_str = form.get("cgpa", "").strip()
        percentage_str = form.get("percentage", "").strip()
        family_income_str = form.get("family_income", "").strip()
        
        disability_val = form.get("disability", "No").strip().lower() in ["yes", "true", "1"]
        disability_pct_str = form.get("disability_percentage", "0").strip()
        bpl_val = form.get("bpl_status", "No").strip().lower() in ["yes", "true", "1"]
        hostel_val = form.get("hostel_status", "No").strip().lower() in ["yes", "true", "1"]

        # Validate required inputs
        if not gender or not domicile or not category or not branch:
            flash("Please fill in all required categorical fields.", "danger")
            return redirect(url_for("questionnaire"))

        try:
            age = int(age_str)
            year = int(year_str)
            cgpa = float(cgpa_str)
            percentage = float(percentage_str)
            family_income = int(family_income_str)
            disability_percentage = int(disability_pct_str) if disability_val else 0
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid numeric input provided: {e}")
            flash("Please provide valid numeric values for Age, Year, CGPA, Percentage, and Income.", "danger")
            return redirect(url_for("questionnaire"))

        # Value bounds validation
        if not (15 <= age <= 60):
            flash("Please enter a valid age between 15 and 60.", "warning")
            return redirect(url_for("questionnaire"))

        if not (1 <= year <= 5):
            flash("Please select a valid academic year (1 to 5).", "warning")
            return redirect(url_for("questionnaire"))

        if not (0.0 <= cgpa <= 10.0):
            flash("CGPA must be between 0.00 and 10.00.", "warning")
            return redirect(url_for("questionnaire"))

        if not (0.0 <= percentage <= 100.0):
            flash("Percentage must be between 0.00 and 100.00%.", "warning")
            return redirect(url_for("questionnaire"))

        if family_income < 0:
            flash("Family income cannot be negative.", "warning")
            return redirect(url_for("questionnaire"))

        # Build clean student profile dictionary (stores < 300 bytes in session cookie)
        student_profile = {
            "gender": gender,
            "age": age,
            "domicile": domicile,
            "category": category,
            "disability": disability_val,
            "disability_percentage": disability_percentage,
            "branch": branch,
            "year": year,
            "cgpa": round(cgpa, 2),
            "percentage": round(percentage, 2),
            "family_income": family_income,
            "bpl_status": bpl_val,
            "hostel_status": hostel_val,
        }

        # Store profile in session (lightweight and secure)
        session["student_profile"] = student_profile

        return redirect(url_for("results"))

    except Exception as e:
        logger.error(f"Unexpected error during recommendation processing: {e}", exc_info=True)
        flash("An unexpected error occurred while processing your profile. Please try again.", "danger")
        return redirect(url_for("questionnaire"))

@app.route("/results")
def results():
    """Displays scholarship recommendation results for the current session."""
    student_profile = session.get("student_profile")

    if not student_profile:
        flash("Please complete the questionnaire first to view your recommendations.", "info")
        return redirect(url_for("questionnaire"))

    # Step 1: Run Machine Learning predictions for supported scholarships
    ai_scores = {}
    try:
        ai_scores = predict_scholarships(student_profile)
    except Exception as e:
        logger.error(f"AI Model prediction encountered error in /results: {e}", exc_info=True)

    # Step 2: Evaluate hard eligibility and hybrid ranking
    results_data = evaluate_and_rank_scholarships(student_profile, ai_scores)

    eligible_scholarships = results_data.get("eligible", [])
    ineligible_scholarships = results_data.get("ineligible", [])
    total_evaluated = results_data.get("total_evaluated", 0)

    # Calculate summary metrics
    total_potential_benefit = sum(
        s.get("maximum_amount") or 0 for s in eligible_scholarships
    )

    high_match_count = sum(
        1 for s in eligible_scholarships if (s.get("recommendation_score_pct") or 0) >= 80
    )

    return render_template(
        "results.html",
        student_profile=student_profile,
        eligible=eligible_scholarships,
        ineligible=ineligible_scholarships,
        total_evaluated=total_evaluated,
        total_potential_benefit=total_potential_benefit,
        high_match_count=high_match_count
    )

@app.route("/scholarships")
def scholarships_list():
    """Searchable directory of all available scholarships."""
    all_scholarships = get_scholarships()
    
    # Extract distinct filter options
    categories = sorted(list(set(
        c for s in all_scholarships for c in s.get("eligible_categories", []) if c != "All"
    )))
    branches = sorted(list(set(
        b for s in all_scholarships for b in s.get("eligible_branches", []) if b != "All"
    )))

    return render_template(
        "scholarships.html",
        scholarships=all_scholarships,
        categories=categories,
        branches=branches
    )

@app.route("/scholarships/<scholarship_id>")
def scholarship_detail(scholarship_id):
    """Detailed view for a specific scholarship."""
    scholarship = get_scholarship_by_id(scholarship_id)
    if not scholarship:
        abort(404)

    # If student has a profile in session, calculate eligibility for this specific scholarship
    student_profile = session.get("student_profile")
    user_status = None
    if student_profile:
        user_status = check_eligibility(student_profile, scholarship)

    return render_template(
        "scholarship_detail.html",
        scholarship=scholarship,
        user_status=user_status,
        student_profile=student_profile
    )

@app.route("/about")
def about():
    """Information regarding the recommendation engine, Machine Learning methodology, and research limitations."""
    return render_template("about.html")

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page."""
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Custom 500 error page."""
    logger.error(f"Internal server error: {e}")
    return render_template("500.html"), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
