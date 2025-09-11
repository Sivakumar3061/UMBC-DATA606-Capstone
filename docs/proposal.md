# 1. UMBC DATA606 Capstone – Vacant Land Price Prediction Proposal

**Project Title:** Spatial Comps for Vacant Land Valuation in Baltimore  
**Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  
**Author:** Sivakumar Deivanayagam  
**GitHub Repo:** https://github.com/Sivakumar3061/UMBC-DATA606-Capstone  
**LinkedIn:** https://www.linkedin.com/in/siva-kumar-deivanayagam/

---

# 2. Background

**What is it about?**  
We aim to predict **vacant land price per square foot** for a target parcel in Baltimore City by leveraging recent nearby **comparable sales (comps)**, parcel characteristics (e.g., lot area), and simple neighborhood indicators. The prediction will be explainable and aligned with how appraisers/realtors think about value.

**Why does it matter?**  
Accurate and transparent valuation supports fair taxation, redevelopment decisions, investor due diligence, and equitable neighborhood revitalization. A data-driven pipeline improves consistency and communication with stakeholders.

**Research questions**
1. How accurately can we predict **price/ft²** for vacant parcels using recent sales plus parcel attributes?
2. How much do **nearby sales features** (distance/time-weighted) improve predictions over a non-spatial baseline?
3. Which factors (lot area, sale recency, neighborhood proxies) most influence **price/ft²**?

---

# 3. Data

**Data sources (provided by student; raw files only):**
- `Real_Property_Information.csv` — Baltimore real property records including sale price/date and assessor attributes.
- `Parcel.csv` — Baltimore parcel table including lot area (`SHAPE_Area`) and join keys (note: CSV export contains no geometry coordinates).

**Data size & shape**
- `Real_Property_Information.csv`: **161.79 MB**, **238,360 rows × 79 columns**
- `Parcel.csv`: **33.99 MB**, **223,652 rows × 15 columns**

**Time period (raw sale date field)**
- `SALEDATE` spans **January 1, 1797** to **August 28, 2025** (historical tail present).
- **Modeling focus:** recent market, e.g., **2019-01-01 to 2025-08-28**.

**What does each row represent?**
- `Real_Property_Information.csv`: one property record with assessor attributes; many rows have sale-related fields (we will treat rows with valid `SALEPRIC` and `SALEDATE` as sales).
- `Parcel.csv`: one parcel record with lot area/perimeter and identifiers.

**Data dictionary (key variables we will use from the raw files)**  
*(Definitions are as used by Baltimore property/parcel data; “Potential values” show examples, not full enumerations.)*

### From `Real_Property_Information.csv`
- **`PIN`** (string) — Parcel Identification Number; will be used to join to `Parcel.csv`.
- **`BLOCKLOT`** (string) — Alternative parcel key (Block–Lot); fallback join key.
- **`BLOCK`** (string/int) — Block number.
- **`LOT`** (string/int) — Lot number.
- **`WARD`** (categorical) — Ward code (e.g., `27`, `26`, `15`, …).
- **`SECTION`** (categorical) — Section code (e.g., `10`, `50`, `20`, …).
- **`NEIGHBOR`** (categorical) — Neighborhood name (e.g., `CANTON`, `BELAIR-EDISON`, `FRANKFORD`, …).
- **`SALEPRIC`** (numeric) — Recorded sale price in USD (target component).
- **`SALEDATE`** (string/int) — Sale date stored as mmddyyyy (e.g., `5222023` → 2023-05-22); will be parsed for time splits/recency.
- **`NO_IMPRV`** (Y/N) — “No improvements” flag; `Y` indicates land-only (examples: `Y`, `N`, missing).
- **`STRUCTAREA`** (numeric) — Building/structure area (ft²); helps confirm land-only when zero.
- **`CURRIMPR`** (numeric) — Assessor’s current improvement value (USD); zero suggests no structure value.
- **`BFCVIMPR`** (numeric) — Base full cash value of improvements (USD).

### From `Parcel.csv`
- **`PIN`** (string) — Parcel key; primary join to `Real_Property_Information.csv`.
- **`BLOCKLOT`** (string) — Alternate parcel key; join fallback if needed.
- **`SHAPE_Area`** (numeric) — Lot area in square feet (from parcel geometry; CSV export includes area, not polygon).
- **`SHAPE_Length`** (numeric) — Parcel perimeter length (feet).
- **`GlobalID`** (string) — Global identifier used by the parcel GIS layer.

**Target/label (to be derived from raw columns)**  
- **`price_per_sqft`** (USD/ft²) = `SALEPRIC` (from Real Property) ÷ `SHAPE_Area` (from Parcel).  
*(Note: We are not adding this to the raw files; it is a derived target computed at modeling time.)*

**Initial feature candidates (from raw columns)**
- Parcel features: `SHAPE_Area` (and transforms like log area), `SHAPE_Length`.
- Sale timing: parsed `SALEDATE` (year/quarter; recency features).
- Location proxies: `WARD`, `SECTION`, `NEIGHBOR`.
- Land-only confirmations: `NO_IMPRV`, `STRUCTAREA`, `CURRIMPR`, `BFCVIMPR`.
- Join keys: `PIN` (primary), `BLOCKLOT` (fallback).

---

# 4. Exploratory Data Analysis (EDA)

**Environment:** Jupyter Notebook (Plotly Express/Matplotlib).

**Steps (using only raw files as inputs):**
1. Parse `SALEDATE` (mmddyyyy → date).
2. Filter **vacant land** heuristically: `NO_IMPRV == 'Y'`, `STRUCTAREA == 0` (or missing), `CURRIMPR == 0`, `BFCVIMPR == 0`.
3. Inner join sales ↔ parcels on `PIN` to access `SHAPE_Area` (lot area).
4. **Target focus:** examine `price_per_sqft` distribution (derived at analysis time), time trends by quarter/year.
5. **Predictor focus:** distributions of `SHAPE_Area` (log-scale), sale recency, and categorical breakdowns across `WARD/SECTION/NEIGHBOR`.
6. **Data quality:** check missing values, obvious outliers in `SALEPRIC`/lot area; deduplicate identical sale records if any.
7. **Augmentation (optional, still raw + public):** simple neighborhood lookup tables or zoning (if allowed) for categorical features.
8. Ensure the final tidy analysis table has one row per **vacant-land sale** with selected predictors and the **derived** target.

---

# 5. Model Training

**Models**
- **Baseline (explainable):** Hedonic regression (OLS) on `log(price_per_sqft)` with `log(SHAPE_Area)`, year fixed effects, and `WARD/SECTION` dummies.
- **Advanced:** Gradient boosting (XGBoost or LightGBM) using the same raw-based features; SHAP for feature attribution.
- **(If geometry is later included):** add true comps features (distance/time-weighted nearby sales) and re-evaluate uplift.

**Train/validation**
- **Temporal split:** Train = 2019–2023; Test = 2024–2025.
- **Cross-validation:** 5-fold within the train period (time-aware).
- **Metrics:** RMSE & MAPE on **price/ft²**; calibration of an 80–90% prediction interval.

**Tech stack**
- Python: `pandas`, `numpy`, `scikit-learn`, `xgboost`/`lightgbm`, `matplotlib`/`plotly`, `shap`.
- Dev: Local Jupyter/VS Code or Google Colab; code versioned on GitHub.

---

# 6. Application of the Trained Models

**Web app (Streamlit recommended)**
- **Input:** Select a parcel (via PIN) or search an address; choose “as-of” date if needed.
- **Output:** Predicted **price/ft²** and implied total price (for the parcel’s area), uncertainty band, and (if geometry is later added) a ranked table of recent nearby comps.
- **Explainability:** Display SHAP feature importances; show how lot area and time effects shape the estimate.

---

# 7. Conclusion

**Summary**  
Using Baltimore’s raw **Real Property** and **Parcel** files, we will build an explainable pipeline to estimate **vacant land price per square foot**. The approach starts with straightforward, reproducible features derived directly from raw columns and can later incorporate spatial comps when geometry is available.

**Limitations**
- Current CSV export for parcels does **not** include polygon coordinates; true distance-based comps and spatial CV require a geospatial export (GeoJSON/Shapefile).
- No explicit arms-length flags; we’ll mitigate with price sanity thresholds and text heuristics (if available).
- Neighborhood effects are coarse without additional context layers.

**Lessons learned & future work**
- With geospatial data, add **distance/time-weighted comps** and amenity distances.
- Incorporate **zoning/land-use** and ACS demographics if allowed by project scope.
- Extend to robust uncertainty estimation and scenario testing (e.g., zoning changes).

---

# 8. References

- Baltimore City **Real Property Information** — raw CSV provided by student.  
- Baltimore City **Parcel** — raw CSV provided by student.  
