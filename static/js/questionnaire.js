/**
 * ScholarMatch - Multi-Step Questionnaire Wizard
 * Vanilla JavaScript (No frameworks)
 */

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("questionnaire-form");
  if (!form) return;

  let currentStep = 1;
  const totalSteps = 5;

  const btnNext = document.getElementById("btn-next");
  const btnPrev = document.getElementById("btn-prev");
  const btnSubmit = document.getElementById("btn-submit");
  const stepItems = document.querySelectorAll(".step-item");
  const formSteps = document.querySelectorAll(".form-step");

  // Conditional Disability Percentage field
  const disabilityRadios = document.querySelectorAll('input[name="disability"]');
  const disabilityPctGroup = document.getElementById("disability-pct-group");
  const disabilityPctInput = document.getElementById("disability_percentage");

  function toggleDisabilityField() {
    const isDisabilityYes = document.querySelector('input[name="disability"]:checked')?.value === "Yes";
    if (disabilityPctGroup) {
      if (isDisabilityYes) {
        disabilityPctGroup.style.display = "flex";
        if (disabilityPctInput) {
          disabilityPctInput.required = true;
          if (!disabilityPctInput.value || parseInt(disabilityPctInput.value) <= 0) {
            disabilityPctInput.value = "40";
          }
        }
      } else {
        disabilityPctGroup.style.display = "none";
        if (disabilityPctInput) {
          disabilityPctInput.required = false;
          disabilityPctInput.value = "0";
        }
      }
    }
  }

  disabilityRadios.forEach((radio) => {
    radio.addEventListener("change", toggleDisabilityField);
  });
  toggleDisabilityField();

  // Validate inputs for current step
  function validateCurrentStep() {
    const activeStepEl = document.querySelector(`.form-step[data-step="${currentStep}"]`);
    if (!activeStepEl) return true;

    const requiredInputs = activeStepEl.querySelectorAll("input[required], select[required]");
    let isValid = true;
    let firstInvalidInput = null;

    // Reset old validation errors
    activeStepEl.querySelectorAll(".input-error-msg").forEach((el) => el.remove());
    activeStepEl.querySelectorAll(".form-control").forEach((el) => {
      el.style.borderColor = "";
    });

    requiredInputs.forEach((input) => {
      // Check radio groups
      if (input.type === "radio") {
        const groupName = input.name;
        const checked = activeStepEl.querySelector(`input[name="${groupName}"]:checked`);
        if (!checked) {
          isValid = false;
          if (!firstInvalidInput) firstInvalidInput = input;
        }
      } else {
        const val = input.value.trim();
        if (!val) {
          isValid = false;
          markInputInvalid(input, "This field is required.");
          if (!firstInvalidInput) firstInvalidInput = input;
        } else {
          // Specific range validation
          if (input.id === "age") {
            const ageVal = parseInt(val);
            if (isNaN(ageVal) || ageVal < 15 || ageVal > 60) {
              isValid = false;
              markInputInvalid(input, "Please enter a valid age between 15 and 60.");
              if (!firstInvalidInput) firstInvalidInput = input;
            }
          }
          if (input.id === "cgpa") {
            const cgpaVal = parseFloat(val);
            if (isNaN(cgpaVal) || cgpaVal < 0 || cgpaVal > 10.0) {
              isValid = false;
              markInputInvalid(input, "CGPA must be between 0.00 and 10.00.");
              if (!firstInvalidInput) firstInvalidInput = input;
            }
          }
          if (input.id === "percentage") {
            const pctVal = parseFloat(val);
            if (isNaN(pctVal) || pctVal < 0 || pctVal > 100.0) {
              isValid = false;
              markInputInvalid(input, "Percentage must be between 0.00 and 100.00%.");
              if (!firstInvalidInput) firstInvalidInput = input;
            }
          }
          if (input.id === "family_income") {
            const incVal = parseInt(val);
            if (isNaN(incVal) || incVal < 0) {
              isValid = false;
              markInputInvalid(input, "Family income cannot be negative.");
              if (!firstInvalidInput) firstInvalidInput = input;
            }
          }
          if (input.id === "disability_percentage" && input.required) {
            const disVal = parseInt(val);
            if (isNaN(disVal) || disVal < 1 || disVal > 100) {
              isValid = false;
              markInputInvalid(input, "Disability % must be between 1% and 100%.");
              if (!firstInvalidInput) firstInvalidInput = input;
            }
          }
        }
      }
    });

    if (!isValid && firstInvalidInput) {
      firstInvalidInput.focus();
    }

    return isValid;
  }

  function markInputInvalid(input, message) {
    input.style.borderColor = "var(--rose-500)";
    const err = document.createElement("span");
    err.className = "input-error-msg";
    err.style.color = "var(--rose-500)";
    err.style.fontSize = "0.8rem";
    err.style.marginTop = "0.25rem";
    err.textContent = message;
    input.parentNode.appendChild(err);
  }

  // Populate Review / Confirmation box in Step 5
  function updateReviewBox() {
    const revGender = document.getElementById("rev-gender");
    const revAge = document.getElementById("rev-age");
    const revDomicile = document.getElementById("rev-domicile");
    const revBranch = document.getElementById("rev-branch");
    const revYear = document.getElementById("rev-year");
    const revCgpa = document.getElementById("rev-cgpa");
    const revPercentage = document.getElementById("rev-percentage");
    const revIncome = document.getElementById("rev-income");
    const revCategory = document.getElementById("rev-category");
    const revDisability = document.getElementById("rev-disability");
    const revBpl = document.getElementById("rev-bpl");
    const revHostel = document.getElementById("rev-hostel");

    if (revGender) revGender.textContent = document.querySelector('input[name="gender"]:checked')?.value || "—";
    if (revAge) revAge.textContent = (document.getElementById("age")?.value || "—") + " years";
    if (revDomicile) revDomicile.textContent = document.getElementById("domicile")?.value || "—";
    if (revBranch) revBranch.textContent = document.getElementById("branch")?.value || "—";
    if (revYear) revYear.textContent = "Year " + (document.querySelector('input[name="year"]:checked')?.value || "—");
    if (revCgpa) revCgpa.textContent = document.getElementById("cgpa")?.value || "—";
    if (revPercentage) revPercentage.textContent = (document.getElementById("percentage")?.value || "—") + "%";
    
    const incomeVal = document.getElementById("family_income")?.value;
    if (revIncome) {
      if (incomeVal) {
        revIncome.textContent = "₹" + parseInt(incomeVal).toLocaleString("en-IN");
      } else {
        revIncome.textContent = "—";
      }
    }

    if (revCategory) revCategory.textContent = document.getElementById("category")?.value || "—";

    const isDis = document.querySelector('input[name="disability"]:checked')?.value === "Yes";
    if (revDisability) {
      revDisability.textContent = isDis ? `Yes (${document.getElementById("disability_percentage")?.value || 40}%)` : "No";
    }

    if (revBpl) revBpl.textContent = document.querySelector('input[name="bpl_status"]:checked')?.value || "No";
    if (revHostel) revHostel.textContent = document.querySelector('input[name="hostel_status"]:checked')?.value || "No";
  }

  // Update step visibility and UI indicators
  function showStep(stepNumber) {
    formSteps.forEach((step) => {
      const stepIndex = parseInt(step.getAttribute("data-step"));
      step.classList.toggle("active", stepIndex === stepNumber);
    });

    stepItems.forEach((item) => {
      const stepIndex = parseInt(item.getAttribute("data-step"));
      item.classList.toggle("active", stepIndex === stepNumber);
      item.classList.toggle("completed", stepIndex < stepNumber);
    });

    // Button states
    btnPrev.style.display = stepNumber === 1 ? "none" : "inline-flex";

    if (stepNumber === totalSteps) {
      btnNext.style.display = "none";
      btnSubmit.style.display = "inline-flex";
      updateReviewBox();
    } else {
      btnNext.style.display = "inline-flex";
      btnSubmit.style.display = "none";
    }

    // Scroll wizard container into view smoothly
    document.querySelector(".wizard-container")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Event Listeners for Next / Back / Submit
  btnNext.addEventListener("click", function () {
    if (validateCurrentStep()) {
      if (currentStep < totalSteps) {
        currentStep++;
        showStep(currentStep);
      }
    }
  });

  btnPrev.addEventListener("click", function () {
    if (currentStep > 1) {
      currentStep--;
      showStep(currentStep);
    }
  });

  form.addEventListener("submit", function (e) {
    if (!validateCurrentStep()) {
      e.preventDefault();
      return;
    }
    // Show loading state
    btnSubmit.disabled = true;
    btnSubmit.innerHTML = `
      <span style="display:inline-block; animation: spin 1s linear infinite; margin-right: 0.5rem;">⟳</span>
      Analyzing Profile & Matching Scholarships...
    `;
  });

  // Initialize
  showStep(currentStep);
});
