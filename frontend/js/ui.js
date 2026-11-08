/**
 * Days to Decay UI Manager
 * Handles DOM rendering, theme management, animations, and history persistence.
 */

const HISTORY_STORAGE_KEY = "days_to_decay_prediction_history";
const THEME_STORAGE_KEY = "days_to_decay_theme";

export const ui = {
  /**
   * Theme toggling and initial state restoration
   */
  initTheme() {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) || "dark";
    this.setTheme(savedTheme);

    const toggleBtn = document.getElementById("themeToggleBtn");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        this.setTheme(nextTheme);
      });
    }
  },

  setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
    const toggleBtn = document.getElementById("themeToggleBtn");
    if (toggleBtn) {
      toggleBtn.innerHTML = theme === "dark" ? "☀️" : "🌙";
      toggleBtn.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
    }
  },

  /**
   * Displays loading state on results card
   */
  showLoading(isLoading, message = "Analyzing Produce & Estimating Degradation...") {
    const overlay = document.getElementById("loadingOverlay");
    const loadingText = document.getElementById("loadingText");
    const analyzeBtn = document.getElementById("analyzeBtn");

    if (overlay) {
      overlay.style.display = isLoading ? "flex" : "none";
      if (loadingText) loadingText.textContent = message;
    }
    if (analyzeBtn) {
      analyzeBtn.disabled = isLoading;
      analyzeBtn.textContent = isLoading ? "Running Inference..." : "Estimate Shelf Life";
    }
  },

  /**
   * Smoothly animates numerical counter
   */
  animateCounter(element, targetValue, duration = 800) {
    if (!element) return;
    const startValue = 0.0;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1.0);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = startValue + (targetValue - startValue) * ease;

      element.textContent = current.toFixed(1);

      if (progress < 1.0) {
        requestAnimationFrame(update);
      } else {
        element.textContent = targetValue.toFixed(1);
      }
    }

    requestAnimationFrame(update);
  },

  /**
   * Updates circular progress indicator (scaled against nominal ~10 days)
   */
  updateProgressRing(remainingDays, maxNominalDays = 10.0) {
    const circle = document.getElementById("progressRingBar");
    if (!circle) return;

    const radius = circle.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    circle.style.strokeDasharray = `${circumference} ${circumference}`;

    const fraction = Math.max(0, Math.min(1.0, remainingDays / maxNominalDays));
    const offset = circumference - fraction * circumference;
    circle.style.strokeDashoffset = offset;

    // Color gradient based on urgency
    if (remainingDays <= 1.0) {
      circle.style.stroke = "var(--accent-rose)";
    } else if (remainingDays <= 3.0) {
      circle.style.stroke = "var(--accent-amber)";
    } else {
      circle.style.stroke = "var(--accent-teal)";
    }
  },

  /**
   * Renders the 5-step aging stage timeline and highlights active stage
   */
  renderTimeline(category, remainingDays) {
    const container = document.getElementById("timelineStagesContainer");
    const progressBar = document.getElementById("timelineProgress");
    if (!container) return;

    const stagesDef = [
      { id: "fresh", label: "Fresh", minDays: category === "banana" ? 7.0 : 5.0 },
      { id: "early", label: "Early Aging", minDays: category === "banana" ? 4.0 : 3.0 },
      { id: "mid", label: "Mid Aging", minDays: category === "banana" ? 2.0 : 1.5 },
      { id: "late", label: "Late Aging", minDays: category === "banana" ? 0.5 : 0.5 },
      { id: "spoiled", label: "Spoiled", minDays: 0.0 },
    ];

    let activeIndex = stagesDef.length - 1;
    for (let i = 0; i < stagesDef.length; i++) {
      if (remainingDays >= stagesDef[i].minDays) {
        activeIndex = i;
        break;
      }
    }

    // Set progress bar width
    if (progressBar) {
      const pct = (activeIndex / (stagesDef.length - 1)) * 100;
      progressBar.style.width = `${pct}%`;
    }

    container.innerHTML = stagesDef
      .map((st, idx) => {
        let stateClass = "";
        if (idx === activeIndex) stateClass = "active";
        else if (idx < activeIndex) stateClass = "passed";

        return `
          <div class="stage-step ${stateClass}">
            <div class="step-node">${idx + 1}</div>
            <div class="step-label">${st.label}</div>
          </div>
        `;
      })
      .join("");
  },

  /**
   * Renders full prediction response to dashboard
   */
  displayPrediction(data, previewDataUrl = null) {
    const emptyState = document.getElementById("emptyResultsState");
    const activeResults = document.getElementById("activeResults");

    if (emptyState) emptyState.style.display = "none";
    if (activeResults) activeResults.style.display = "flex";

    // Animated Days Counter
    const daysDigits = document.getElementById("predictedDaysDigits");
    this.animateCounter(daysDigits, data.predicted_remaining_days);

    // Progress Ring
    const nominal = data.food_category === "banana" ? 10.0 : data.food_category === "avocado" ? 8.0 : 6.0;
    this.updateProgressRing(data.predicted_remaining_days, nominal);

    // Category Pill
    const catEl = document.getElementById("resultCategory");
    if (catEl) catEl.textContent = data.food_category.toUpperCase();

    // Aging Stage & Visual Condition
    const stageEl = document.getElementById("resultStage");
    if (stageEl) stageEl.textContent = data.aging_stage;

    const condEl = document.getElementById("resultCondition");
    if (condEl) condEl.textContent = data.condition;

    // Uncertainty Bounds
    const uncertEl = document.getElementById("resultUncertainty");
    if (uncertEl) {
      if (data.lower_bound_days !== null && data.upper_bound_days !== null) {
        uncertEl.textContent = `${data.lower_bound_days} – ${data.upper_bound_days} days`;
      } else {
        uncertEl.textContent = "Uncalibrated (± estimate)";
      }
    }

    const uncertSubEl = document.getElementById("resultUncertaintyMethod");
    if (uncertSubEl) {
      uncertSubEl.textContent = data.uncertainty_method || "Inductive Conformal Prediction";
    }

    // Timeline
    this.renderTimeline(data.food_category, data.predicted_remaining_days);

    // Save to local history
    this.saveHistory({
      category: data.food_category,
      remainingDays: data.predicted_remaining_days,
      stage: data.aging_stage,
      thumbnail: previewDataUrl,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    });
  },

  /**
   * Local history management
   */
  saveHistory(item) {
    try {
      const history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
      history.unshift(item);
      const trimmed = history.slice(0, 6); // keep 6 recent
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(trimmed));
      this.renderHistory();
    } catch (e) {
      console.warn("Could not save history to localStorage:", e);
    }
  },

  renderHistory() {
    const grid = document.getElementById("historyGrid");
    if (!grid) return;

    try {
      const history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
      if (history.length === 0) {
        grid.innerHTML = `<p style="grid-column: 1/-1; color: var(--text-muted); font-size: 0.9rem;">No recent predictions recorded on this browser yet.</p>`;
        return;
      }

      grid.innerHTML = history
        .map((h) => `
          <div class="history-card">
            ${h.thumbnail ? `<img class="history-thumb" src="${h.thumbnail}" alt="Sample" />` : `<div class="history-thumb" style="background:var(--bg-tertiary); display:flex; align-items:center; justify-content:center;">🍏</div>`}
            <div class="history-meta">
              <div class="history-cat">${h.category}</div>
              <div class="history-days">${h.remainingDays}d <span style="font-size:0.75rem; font-weight:normal; color:var(--text-muted)">left</span></div>
              <div class="history-date">${h.stage} · ${h.timestamp}</div>
            </div>
          </div>
        `)
        .join("");
    } catch (e) {
      console.warn("Could not render history:", e);
    }
  },

  showError(message) {
    alert(`Prediction Notice: ${message}`);
  },
};

