# DAYS TO DECAY — Dataset Engineering & Image Collection Guide

## 1. Overview and Scientific Motivation

Unlike object classification datasets (which simply classify an image as "banana" or "bread"), **DAYS TO DECAY** trains a continuous-value regression model to estimate the remaining shelf life ($Y \in \mathbb{R}^+$) of perishable goods.

Because perishable deterioration is a continuous biological process influenced by moisture loss, enzymatic browning, respiration rate, and fungal spore emergence, high dataset fidelity and strict split isolation are paramount.

---

## 2. Supported Categories & Spoilage Endpoints

A consistent, objective spoilage endpoint must be defined prior to study initiation. Do not silently mix different spoilage definitions.

| Perishable Category | Nominal Shelf Life (Room Temp) | Objective Spoilage Endpoint Definition |
|---|---|---|
| **Banana** (*Musa acuminata*) | ~8 to 11 days | **Stage 7+ Senescent Collapse:** Over 70% dark brown/black necrotic skin coverage, structural pulp liquefaction/collapse, or mycelium emergence at stem or blossom end, rendering fruit unpalatable. |
| **Avocado** (*Persea americana*) | ~6 to 9 days | **Indentation Collapse & Internal Rancidity:** Total yielding under gentle palm pressure, skin shrinkage/collapse, rancid fermentation odor, internal vascular stringiness/darkening, or visible fungal colonies. |
| **Bread** (White / Sourdough Loaf) | ~5 to 7 days | **Fungal Sporulation / Staling Petrification:** Macroscopic pinpoint colonies of bread mold (*Rhizopus stolonifer*, *Penicillium*, *Aspergillus*), or complete moisture loss causing unyielding starch retrogradation. |

---

## 3. Standardization and Photography Protocol

To minimize confounding noise from varying lighting or angles, adhere to this capture protocol:

### Physical Setup
- **Lighting:** Diffuse daylight or a fixed 5000K daylight-balanced LED lamp. Avoid direct, harsh flash reflections that wash out subtle browning spots.
- **Background:** Uniform neutral matte background (white, light gray, or kraft paper). Avoid patterned tablecloths.
- **Camera Angle:** Fixed 45-degree oblique angle showing both top surface and lateral profile.
- **Distance:** Approximately 30 cm (12 inches) from the specimen, centering the item within the frame.
- **Capture Cadence:** Exactly once every 24 hours at the same time of day (e.g., 10:00 AM).

### Data Schema Fields
Each sample row in `data/metadata/dataset.csv` contains:

```csv
image_path,food_category,sample_id,capture_date,capture_day,initial_weight,temperature,humidity,initial_freshness,observed_spoilage_date,remaining_days,storage_condition,collection_notes
```

- `sample_id`: Unique identifier for each physical specimen (e.g. `BANANA_001`, `BANANA_002`, `AVOCADO_001`).
- `capture_day`: Integer counter starting at Day 0 on initiation day.
- `observed_spoilage_date`: Final verified spoilage date (YYYY-MM-DD).
- `remaining_days`: Computed ground-truth target:
  $$\text{remaining\_days} = \text{observed\_spoilage\_date} - \text{capture\_date}$$

---

## 4. Preventing Data Leakage: Group-Based Splitting

> [!IMPORTANT]
> **Zero Data Leakage Requirement:**
> All time-series photographs of a single physical specimen (e.g. `BANANA_001` on Day 0, Day 1, Day 2... Day 8) **MUST** remain exclusively within the same train, validation, or test split.
> 
> Randomly assigning individual photographs from the same specimen across different splits allows the model to memorize specimen-specific blemishes (bruises, stem shapes), causing catastrophic data leakage and falsely inflating evaluation metrics.

Our pipeline strictly enforces this via `GroupShuffleSplit(groups=df['sample_id'])`.

---

## 5. Dataset Collection & Validation CLI Tools

### Collecting a Photograph
Use the interactive or automated collection tool:

```bash
# Automated registration from a smartphone photo
python scripts/collect_dataset.py \
  --image /path/to/phone_photo.jpg \
  --category banana \
  --sample-id BANANA_001 \
  --capture-date 2026-11-08 \
  --capture-day 0 \
  --spoilage-date 2026-11-18 \
  --storage room_temperature \
  --notes "Cavendish banana, ambient 21C"

# Interactive wizard mode
python scripts/collect_dataset.py --interactive
```

### Preprocessing to Standard 224x224 RGB
Once photos are cataloged in `data/raw/`, run preprocessing to resize and crop:

```bash
python scripts/preprocess_dataset.py
```

### Validating Dataset Integrity
Run the diagnostic suite to verify file existence, valid dimensions, non-negative targets, and consistent spoilage dates:

```bash
python scripts/validate_dataset.py --strict
```
A markdown report is automatically generated at `reports/metrics/dataset_validation_report.md`.

