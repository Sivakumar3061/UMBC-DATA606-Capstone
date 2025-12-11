# Spatial Comps for Vacant Land Valuation in Baltimore
### UMBC Data Science Master’s Degree Capstone  
Instructor: **Dr. Chaojie (Jay) Wang**

---

## 1. Title and Author

**Project Title:** Spatial Comps for Vacant Land Valuation in Baltimore  
**Author:** *Sivakumar Deivanayagam*  

**GitHub Repository:** *https://github.com/Sivakumar3061/UMBC-DATA606-Capstone*  
**LinkedIn Profile:** *https://www.linkedin.com/in/siva-kumar-deivanayagam/*  
**PowerPoint Presentation:** *https://drive.google.com/drive/folders/15EjSl8UmnekECu-Y2jvmo9L3e-XVS064?usp=sharing*  
**YouTube Video:** *(insert link if applicable)*  

---

## 2. Background

Baltimore City lacks a consistent and transparent methodology for valuing vacant land parcels. Manual appraisal varies widely across neighborhoods, and valuation inconsistency affects taxation fairness, urban development, and real-estate investment decisions.

### **What is this project about?**  
This project develops a **machine learning–driven land valuation system** utilizing property sales, parcel geometry, and spatial comparable sales to estimate **price per square foot (PPSF)** for vacant land.

### **Why does it matter?**

- Ensures **equitable taxation**  
- Supports **redevelopment and city planning**  
- Provides **investors** with reliable valuation tools  
- Offers **policy-makers** data-driven insights  
- Standardizes a process that is traditionally subjective  

### **Research Questions**

1. Can spatial features (comps, distance metrics) significantly improve valuation accuracy?  
2. Do KNN-based comparable sales explain parcel price better than traditional features?  
3. Which parcel attributes and spatial factors most influence PPSF?  
4. Can the final model be operationalized via a Streamlit web app for public use?

---

## 3. Data

### **Data Sources**

| Dataset | Description |
|--------|-------------|
| **Real_Property_Information.csv** | Historical Baltimore property sales (2019–2025) |
| **Parcel.csv / Parcel.geojson** | Parcel geometry, lot area, spatial identifiers |

### **Data Size**

- Total records: **~238,000 sales**  
- Parcel dataset: **~200,000 parcels**  
- Combined working dataset: **~300-400 MB**

### **Data Shape**

- Property dataset: ~238k × 50+ columns  
- Parcel dataset: ~200k × 20 columns  

### **Time Period**  
Sales from **2019 to 2025**

### **Row Definition**

Each cleaned row represents **one vacant land sale event**.

---

### **Data Dictionary (Key Columns)**

| Column | Type | Definition |
|--------|------|------------|
| `PIN` | String | Unique parcel ID |
| `BLOCKLOT` | String | Block + lot composite key |
| `SALEPRIC` | Numeric | Final transaction price |
| `SALEDATE` | Date | Sale date |
| `NO_IMPRV` | Int | 1 if vacant land |
| `STRUCTAREA` | Int | Structure area (0 for vacant) |
| `SHAPE_Area` | Float | Lot area in sq. ft. |
| `NEIGHBOR` | Categorical | Neighborhood name |
| `lat`, `lon` | Float | Parcel centroid |
| `price_per_sqft` | Float | Target variable |
| `knn_mean_ppsf` | Float | Mean PPSF of 5 nearest comps |
| `spatial_lag_ppsf` | Float | Distance-weighted PPSF |

### **Target Variable**
`price_per_sqft = SALEPRIC / SHAPE_Area`

### **Candidate Features**

- Lot size (log-transformed)  
- Neighborhood indicators  
- Ward, Section  
- Year, quarter  
- KNN comparable sales  
- Spatial lag penalties  
- Distance to city center  
- KMeans clusters  

---

## 4. Exploratory Data Analysis (EDA)

Performed in Jupyter Notebook using **Pandas, GeoPandas, Plotly, and Matplotlib**.

### **Cleaning Steps**

- Filtered vacant parcels via `NO_IMPRV` and `STRUCTAREA = 0`  
- Removed **10,000+ duplicates**  
- Joined datasets using `PIN` and fallback `BLOCKLOT`  
- Extracted `year`, `quarter` from `SALEDATE`  
- Log-transformed PPSF  
- Computed spatial features via KNN and distance matrices  

### **Missing Values**

- Minor SHAPE_Area missing values → removed  
- Missing neighborhoods → labeled or dropped  

### **Summary of Findings**

- PPSF is right-skewed  
- Strong neighborhood clustering  
- Spatial dependence is significant (supports comps logic)

### **Potential Augmentations**

- Zoning data  
- Economic indicators  
- Population and crime statistics  

### **Tidy Dataset Criteria Achieved**

- Each row = **one parcel sale**  
- Each column = **one attribute**  
- Clean unique identifiers  

---

## 5. Model Training

Two predictive models were developed:

### **Model 1 — OLS Hedonic Regression**

- Predicts **log(PPSF)**  
- Features: lot size, WARD, SECTION, NEIGHBOR, year  
- Compares traditional appraisal-style valuation  
- Strength: interpretable  
- Weakness: cannot capture nonlinear spatial complexity  

### **Model 2 — Spatial XGBoost**

Incorporates:

- KNN comparable sales  
- Spatial lag PPSF  
- Distance metrics  
- Neighborhood/time one-hot encodings  
- KMeans clusters  

### **Train/Test Split**
`80/20 split`

### **Model Performance**

| Model | R² | Notes |
|-------|------|-------|
| **OLS** | ~0.60 | Moderate |
| **XGBoost** | ~0.94 (log space) | Excellent spatial capture |

Median Absolute Percentage Error (MdAPE): **< 20%**

### **Top SHAP Features**

- Lot size (log_area)  
- KNN mean PPSF  
- Spatial lag PPSF  
- Neighborhood fixed effects  
- Temporal variables (year/quarter)  

### **Tools & Libraries**

- pandas, numpy  
- scikit-learn  
- XGBoost  
- GeoPandas  
- SHAP  
- Plotly Express  

---

## 6. Application of the Trained Models

A **Streamlit web app** was built to democratize parcel valuation.

### **User Features**

- Search parcels by **PIN** or neighborhood  
- Real-time PPSF and property valuation  
- Automatic comparable sales (KNN)  
- Transparent SHAP-based explanations  

### **Tech Stack**

- **Streamlit** for UI  
- **XGBoost** for ML  
- **GeoPandas** for spatial processing  
- **Pandas** for pipeline operations  

---

## 7. Conclusion

### **Summary**

This project:

- Processed and cleaned complex real-estate datasets  
- Engineered spatially intelligent features  
- Demonstrated large accuracy improvements using spatial ML  
- Delivered a working valuation model and interactive app  

### **Limitations**

- Missing detailed parcel polygons  
- No arms-length sale indicator  
- Some neighborhoods have sparse transactions  
- Lacks zoning/economic features  

### **Lessons Learned**

- Spatial data drastically improves real-estate modeling  
- Data wrangling is the most intensive stage  
- SHAP is essential for interpretability  
- Deployment skills are critical for real-world DS work  

### **Future Work**

- Integrate zoning + demographic data  
- Use full parcel geometry for distance calculations  
- Build uncertainty intervals  
- Deploy full API backend and stable cloud app  

---

## 8. References

- Baltimore City Open Data Portal  
- GeoPandas Documentation – https://geopandas.org  
- XGBoost Documentation – https://xgboost.readthedocs.io  
- SHAP Documentation – https://shap.readthedocs.io  
- Spatial comps and modeling details from project notebook & presentation  

---
