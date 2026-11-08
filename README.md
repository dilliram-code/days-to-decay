# DAYS TO DECAY 🍃
### Continuous Perishable Food Aging & Remaining Shelf-Life Estimation System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"See the aging. Estimate the remaining life."**  
> An end-to-end computer vision and uncertainty-quantified deep learning system estimating the continuous remaining shelf life of perishable produce and bread from single photographs.

---

## 1. Project Overview & Vision

Traditional computer vision applications classify static objects (e.g., *"this is a banana"*). In contrast, **DAYS TO DECAY** approaches food longevity as a continuous visual regression problem:

$$\text{Input: Single RGB Photograph} \quad \xrightarrow{\quad \text{ResNet50 Backbone} \quad} \quad \text{Output: Remaining Days } (\hat{y} \in \mathbb{R}^+)$$

The network learns subtle visual indicators of degradation over time:
- **Enzymatic browning & chlorophyll breakdown** (green to bright yellow to necrotic brown spotting).
- **Surface dehydration, wrinkling, and skin shrinkage**.
- **Pulp softening & structural indentation collapse**.
- **Initial mycelial emergence & mold colonization**.

### Core Perishable Categories
1. **Banana** (*Musa acuminata*): Nominal ~10 days shelf life.
2. **Avocado** (*Persea americana*): Nominal ~8 days shelf life.
3. **Bread** (Loaf Crumb/Crust): Nominal ~6 days shelf life.

---

## 2. Key Engineering Highlights

- 🧠 **Continuous Regression Head:** Deep feature extraction via pretrained ResNet50 (and EfficientNet-B0), replacing the classification softmax with global pooling, dropout, linear projection, and a **ReLU activation** ensuring physically meaningful non-negative shelf-life estimates ($\hat{y} \ge 0.0$).
- 🛡️ **Zero-Leakage Group Splitting:** Prevents catastrophic data leakage by keeping all time-series captures of a physical specimen (`sample_id`) strictly inside a single split (Train, Val, or Test).
- 📊 **Calibrated Uncertainty (Conformal Prediction):** Avoids fake confidence intervals. Uses distribution-free **Inductive Conformal Prediction** on held-out validation residuals to return verified 90% finite-sample prediction intervals.
- ⚡ **Production FastAPI Service:** Modular async backend with file verification, MIME checks, model singleton caching, batch inference, and unified static frontend hosting.
- 🎨 **Modern Vanilla Web Interface:** Responsive dark/light theme, glassmorphism design, SVG progress ring, dynamic aging stage timeline, drag-and-drop workspace, and localStorage session history without heavyweight JavaScript frameworks.
- 🧪 **100% Automated Testing:** Complete test suite using `pytest` covering dataset loading, transforms, architecture constraints, non-negativity, and API routes.

---

## 3. System Architecture

```
   +-------------------------------------------------------------+
   |                    RAW DATA INGESTION                       |
   |  Smartphone Photos / Webcam  -->  scripts/collect_dataset.py |
   +-------------------------------------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   |                  PREPROCESSING & VALIDATION                 |
   |  RGB 224x224 Resize, Bilinear Norm, Schema Integrity Checks |
   |  scripts/preprocess_dataset.py | scripts/validate_dataset.py|
   +-------------------------------------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   |               GROUP-BASED SPLITTING (NO LEAKAGE)            |
   |  Strictly partitions sample_id into Train / Val / Test      |
   +-------------------------------------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   |                TRANSFER LEARNING REGRESSION                 |
   |  ResNet50 Backbone -> Linear(2048->256) -> LeakyReLU        |
   |  -> Dropout(0.3) -> Linear(256->1) -> ReLU -> Days          |
   |  Huber / MSE Loss, AdamW Optimizer, Cosine Scheduler        |
   +-------------------------------------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   |            CONFORMAL PREDICTION CALIBRATION                 |
   |  Calculates absolute residual quantile q on validation set  |
   |  Interval: [ max(0, y_pred - q),  y_pred + q ] (90% Conf)   |
   +-------------------------------------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   |            FASTAPI BACKEND & VANILLA FRONTEND               |
   |  REST API (/health, /predict, /predict/batch, /categories)  |
   |  Interactive UI with Animated Meter & Aging Stage Timeline  |
   +-------------------------------------------------------------+
```

---

## 4. Directory Structure

```
days-to-decay/
├── README.md                   # Complete architectural guide
├── DATASET_GUIDE.md            # Daily photography & spoilage protocol
├── MODEL_CARD.md               # Scientific model card & ethics
├── requirements.txt            # Dependency manifest
├── render.yaml                 # Render cloud deployment blueprint
├── .env.example                # Environment settings template
├── configs/
│   └── train_config.yaml       # Hyperparameter and artifact configuration
├── data/
│   ├── raw/                    # Cataloged original photographs
│   ├── processed/              # Standardized 224x224 RGB images
│   └── metadata/
│       └── dataset.csv         # Verified sample-day metadata catalog
├── notebooks/
│   └── exploratory_data_analysis.ipynb # EDA & degradation trajectories
├── scripts/
│   ├── collect_dataset.py      # Specimen intake & camera utility
│   ├── preprocess_dataset.py   # Standardization pipeline
│   ├── validate_dataset.py     # Data integrity diagnostic suite
│   ├── train.py                # Transfer learning training runner
│   ├── evaluate.py             # Held-out evaluation & report generation
│   └── predict.py              # Standalone CLI prediction utility
├── src/
│   ├── data/                   # Dataset loader, transforms, schemas
│   ├── models/                 # ResNet50 regressor & factory
│   ├── training/               # Losses, metrics, trainer loop
│   └── evaluation/             # Evaluator & publication-quality plots
├── backend/
│   └── app/                    # FastAPI routes, schemas, services
├── frontend/                   # Vanilla HTML5, CSS3, ES modules
└── tests/                      # Automated pytest suite
```

---

## 5. Quickstart & Installation

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/days-to-decay.git
cd days-to-decay

# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Test Suite

```bash
pytest tests/ -v
```

All 18 automated tests will run and verify dataset validation, model output constraints, non-negativity, and API endpoints.

---

## 6. End-to-End Operational Workflow

### Step 1: Collect & Catalog Produce Photos
```bash
# Register a newly captured produce photo
python scripts/collect_dataset.py \
  --image sample_banana.jpg \
  --category banana \
  --sample-id BANANA_001 \
  --capture-date 2026-11-08 \
  --capture-day 0 \
  --spoilage-date 2026-11-18 \
  --storage room_temperature
```

### Step 2: Validate Dataset Quality
```bash
python scripts/validate_dataset.py --strict
```
Generates a markdown quality report at `reports/metrics/dataset_validation_report.md`.

### Step 3: Train the Shelf-Life Regressor
```bash
python scripts/train.py --config configs/train_config.yaml
```
- Performs group-based splitting by `sample_id` (zero sample leakage).
- Freezes early backbone layers for the initial epochs, then fine-tunes.
- Saves best checkpoint to `artifacts/best_model.pth`.
- Calibrates 90% conformal prediction bounds to `artifacts/calibration_residuals.json`.

### Step 4: Run Held-Out Model Evaluation
```bash
python scripts/evaluate.py
```
Outputs evaluation metrics and plots to `reports/figures/`:
- `actual_vs_predicted.png` (Scatter with $y=x$ ideal line and $\pm 1.0\text{d}$ band)
- `residual_distribution.png` (KDE error distribution)
- `training_curves.png` (Loss & validation MAE progression)
- `error_by_category.png` (Box plots by produce type)
- `reports/evaluation_report.md` (Markdown benchmark against naive baseline)

### Step 5: Run Standalone CLI Prediction
```bash
python scripts/predict.py --image /path/to/banana.jpg --category banana
```

Example Output:
```text
=== DAYS TO DECAY: Shelf-Life Estimation ===
Category:                 BANANA
Predicted Remaining Life: 2.4 days
90% Prediction Interval:  1.6 - 3.2 days (conformal_prediction)
Aging Stage:              Late Ripening
Visual Condition:         Moderate Browning
Scientific Disclaimer:    This prediction is an empirical visual estimate and does not establish microbiological food safety.
```

---

## 7. Running the Web Application

Launch the unified FastAPI server and frontend:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
```
http://localhost:8000
```

### REST API Endpoints
- `GET /health`: System diagnostics, device status, and checkpoint state.
- `GET /categories`: Supported produce categories, nominal days, and stage criteria.
- `GET /model-info`: Architecture details, input dimensions, and evaluation metrics.
- `POST /predict`: Upload image (`multipart/form-data`) + `food_category` -> JSON estimate.
- `POST /predict/batch`: Process multiple photos sequentially.
- `GET /docs`: Interactive Swagger UI.

---

## 8. Deployment on Render

This repository is pre-configured for one-click deployment on **Render**:

1. Create a **New Web Service** on Render pointing to your repository.
2. Select **Python** environment.
3. Configure:
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
4. Set environment variable: `DEVICE=cpu`.

The application serves the static frontend directly from FastAPI while responding to REST API calls at relative paths (`/predict`, `/health`).

---

## 9. Scientific & Food Safety Disclaimer

> [!CAUTION]
> **IMPORTANT FOOD SAFETY NOTICE:**  
> DAYS TO DECAY estimates remaining shelf life based on visual surface degradation (color, texture, browning) under typical ambient room temperature conditions.  
> 
> Visual appearance **does not establish microbiological safety or the absence of pathogenic bacterial toxins** (e.g. *Salmonella*, *Listeria*, *Staphylococcus aureus*). Always inspect food for off-odors, abnormal softness, sliminess, or internal rot, and discard any food in doubt.

---

## 10. License

Released under the [MIT License](LICENSE).
