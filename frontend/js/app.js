/**
 * Days to Decay Application Controller
 * Handles user interactions, file drop events, form binding, and API execution.
 */

import { api } from "./api.js";
import { ui } from "./ui.js";

let selectedFile = null;
let selectedCategory = "banana";
let previewBase64 = null;

document.addEventListener("DOMContentLoaded", async () => {
  // 1. Initialize UI theme & local history
  ui.initTheme();
  ui.renderHistory();

  // 2. Perform background health check
  checkSystemHealth();

  // 3. Setup File Upload & Drag-and-Drop
  setupDropzone();

  // 4. Setup Category Selection Pills
  setupCategoryPills();

  // 5. Setup Action Buttons
  setupActionButtons();
});

async function checkSystemHealth() {
  const badge = document.getElementById("systemStatusBadge");
  const dot = document.getElementById("systemStatusDot");
  const text = document.getElementById("systemStatusText");

  try {
    const health = await api.getHealth();
    if (health.status === "healthy") {
      if (dot) dot.style.background = "var(--accent-emerald)";
      if (text) {
        text.textContent = health.model_loaded ? "Model Ready" : "Pipeline Active (Base)";
      }
    } else {
      if (dot) dot.style.background = "var(--accent-amber)";
      if (text) text.textContent = "Connecting...";
    }
  } catch (err) {
    if (dot) dot.style.background = "var(--accent-rose)";
    if (text) text.textContent = "Offline";
  }
}

function setupCategoryPills() {
  const pills = document.querySelectorAll(".category-pill");
  pills.forEach((pill) => {
    pill.addEventListener("click", () => {
      pills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      const radio = pill.querySelector("input[type='radio']");
      if (radio) {
        radio.checked = true;
        selectedCategory = radio.value;
      }
    });
  });
}

function setupDropzone() {
  const dropzone = document.getElementById("uploadDropzone");
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage = document.getElementById("previewImage");
  const removeBtn = document.getElementById("removeImageBtn");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  // Drag over / leave
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  // Drop
  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  // File input change
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Remove preview
  if (removeBtn) {
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      selectedFile = null;
      previewBase64 = null;
      fileInput.value = "";
      if (previewContainer) previewContainer.style.display = "none";
      dropzone.style.display = "block";
    });
  }
}

function handleFileSelected(file) {
  const dropzone = document.getElementById("uploadDropzone");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage = document.getElementById("previewImage");

  const validTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!validTypes.includes(file.type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
    alert("Please select a valid image photograph (JPG, PNG, or WEBP).");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    alert("Photograph exceeds 10MB size limit.");
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewBase64 = e.target.result;
    if (previewImage) previewImage.src = previewBase64;
    if (previewContainer) previewContainer.style.display = "block";
    if (dropzone) dropzone.style.display = "none";
  };
  reader.readAsDataURL(file);
}

function setupActionButtons() {
  const analyzeBtn = document.getElementById("analyzeBtn");
  if (!analyzeBtn) return;

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      alert("Please upload or drag-and-drop a photograph of your perishable food item.");
      return;
    }

    if (!selectedCategory) {
      alert("Please select the food category (Banana, Avocado, or Bread).");
      return;
    }

    try {
      ui.showLoading(true, `Analyzing ${selectedCategory.toUpperCase()} Degradation...`);
      const response = await api.predict(selectedFile, selectedCategory);
      ui.displayPrediction(response, previewBase64);
    } catch (err) {
      console.error("Prediction error:", err);
      ui.showError(err.message || "An error occurred during regression inference.");
    } finally {
      ui.showLoading(false);
    }
  });
}
