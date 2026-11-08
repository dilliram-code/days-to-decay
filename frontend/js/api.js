/**
 * Days to Decay API Client
 * Clean fetch wrapper handling backend requests, errors, and timeouts.
 */
const API_BASE_URL = window.API_BASE_URL || "/api";

export const api = {
  /**
   * Health check and diagnostics
   */
  async getHealth() {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      return await res.json();
    } catch (err) {
      console.warn("Backend health check failed:", err.message);
      return { status: "unreachable", error: err.message };
    }
  },

  /**
   * Fetch supported food categories and aging stages
   */
  async getCategories() {
    const res = await fetch(`${API_BASE_URL}/categories`);
    if (!res.ok) throw new Error(`Failed to fetch categories: ${res.status}`);
    return await res.json();
  },

  /**
   * Fetch model architecture & evaluation metadata
   */
  async getModelInfo() {
    const res = await fetch(`${API_BASE_URL}/model-info`);
    if (!res.ok) throw new Error(`Failed to fetch model info: ${res.status}`);
    return await res.json();
  },

  /**
   * Send image file and category to prediction endpoint
   */
  async predict(file, foodCategory) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("food_category", foodCategory);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || data.error || `Server responded with ${res.status}`);
      }
      return data;
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        throw new Error("Prediction request timed out after 30 seconds.");
      }
      throw err;
    }
  },
};
