/**
 * PCCOE Scholarship Portal - Main Application Client Interactions
 * Vanilla JavaScript (No frameworks)
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Results Filter Tabs
  const filterTabs = document.querySelectorAll(".filter-tab");
  const scholarshipCards = document.querySelectorAll(".scholarship-card");

  filterTabs.forEach((tab) => {
    tab.addEventListener("click", function () {
      filterTabs.forEach((t) => t.classList.remove("active"));
      this.classList.add("active");

      const filterValue = this.getAttribute("data-filter");

      scholarshipCards.forEach((card) => {
        const isEligible = card.getAttribute("data-eligible") === "true";
        const score = parseInt(card.getAttribute("data-score") || "0", 10);

        if (filterValue === "all-eligible") {
          card.style.display = isEligible ? "flex" : "none";
        } else if (filterValue === "high-match") {
          card.style.display = isEligible && score >= 80 ? "flex" : "none";
        } else if (filterValue === "ineligible") {
          card.style.display = !isEligible ? "flex" : "none";
        } else if (filterValue === "all") {
          card.style.display = "flex";
        }
      });
    });
  });

  // 2. Directory / Browser Search & Filters
  const searchInput = document.getElementById("search-input");
  const filterBranch = document.getElementById("filter-branch");
  const filterGender = document.getElementById("filter-gender");
  const filterCategory = document.getElementById("filter-category");
  const browserCards = document.querySelectorAll(".browser-card");
  const noResultsEl = document.getElementById("no-scholarships-found");

  function filterDirectory() {
    if (!browserCards.length) return;

    const query = (searchInput?.value || "").toLowerCase().trim();
    const selectedBranch = (filterBranch?.value || "all").toLowerCase();
    const selectedGender = (filterGender?.value || "all").toLowerCase();
    const selectedCategory = (filterCategory?.value || "all").toLowerCase();

    let visibleCount = 0;

    browserCards.forEach((card) => {
      const name = (card.getAttribute("data-name") || "").toLowerCase();
      const provider = (card.getAttribute("data-provider") || "").toLowerCase();
      const branches = (card.getAttribute("data-branches") || "").toLowerCase();
      const genders = (card.getAttribute("data-gender") || "").toLowerCase();
      const categories = (card.getAttribute("data-category") || "").toLowerCase();
      const desc = (card.getAttribute("data-desc") || "").toLowerCase();

      const matchesQuery =
        !query ||
        name.includes(query) ||
        provider.includes(query) ||
        desc.includes(query);

      const matchesBranch =
        selectedBranch === "all" ||
        branches.includes("all") ||
        branches.includes(selectedBranch);

      const matchesGender =
        selectedGender === "all" ||
        genders.includes("all") ||
        genders.includes(selectedGender);

      const matchesCategory =
        selectedCategory === "all" ||
        categories.includes("all") ||
        categories.includes(selectedCategory);

      if (matchesQuery && matchesBranch && matchesGender && matchesCategory) {
        card.style.display = "flex";
        visibleCount++;
      } else {
        card.style.display = "none";
      }
    });

    if (noResultsEl) {
      noResultsEl.style.display = visibleCount === 0 ? "block" : "none";
    }
  }

  if (searchInput) searchInput.addEventListener("input", filterDirectory);
  if (filterBranch) filterBranch.addEventListener("change", filterDirectory);
  if (filterGender) filterGender.addEventListener("change", filterDirectory);
  if (filterCategory) filterCategory.addEventListener("change", filterDirectory);
});
