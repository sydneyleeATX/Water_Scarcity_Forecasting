# Water Scarcity Forecasting: Lake Travis, Austin TX

## Project Objective
This project investigates how machine learning can be applied to forecast water scarcity in the context of Lake Travis, Austin TX.  
With climate change intensifying water availability challenges, the goal was to build a predictive model of water levels that leverages environmental, hydrological, and socio-economic features. The project aims to provide insights that may inform policy and water resource management.

---

## Implementation Approach

### 1. Data Collection & Cleaning
- Collected data from multiple sources:
  - Regional population (census data)  
  - Sectoral water usage  
  - Weather indicators (temperature, precipitation, degree days)  
  - Palmer Drought Severity Index (PDSI)  
  - Standardized Precipitation Index (SPI)  
- Standardized all datasets with `year` and `month` columns.  
- Interpolated missing values (linear interpolation, mean imputation, seasonal scaling).  
- Scaled yearly data into monthly estimates for consistency.  

### 2. Feature Engineering
- Built interaction terms to capture hidden relationships:  
  - `Population * Groundwater`  
  - `Population * Surface Water`  
  - `Precipitation * Irrigation Surface Water`  
  - `Days since precipitation * Irrigation Groundwater`  
  - `Year * PDSI`, `Year * SPI`  
- Created lagged features for water level and weather metrics.  

### 3. Exploratory Data Analysis
- Dataset after cleaning: 936 rows × 66 features.  
- Visualized correlations between features and target (`water_level`).  
- Identified 19 features with |correlation| > 0.15 for further analysis.  
- Scatterplots and time trends highlighted nonlinear patterns and drought-related outliers.  

### 4. Feature Selection
- Recursive Feature Elimination (RFE) performed after initial training.  
- Final feature set included lagged water levels, precipitation, PDSI, SPI, degree days, and engineered interaction terms.  

### 5. Model Building
- XGBoost Regressor selected due to:
  - Ability to handle nonlinear relationships.  
  - Robustness to missing data.  
  - Built-in feature importance metrics.  
- Train/Test split: 80/20 chronological split.  
- Hyperparameter tuning via `GridSearchCV`:
  
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1]
}


## 6. Forecasting Strategy

**Case 1 (Lag Data Available):**  
- MAE = 2.44 | Relative MAE = 0.37% → highly accurate.  

**Case 2 (Lag Data Forecasted via Forward Fill):**  
- MAE = 8.69 | Relative MAE = 1.33% → still accurate in trend replication.  

---

## 7. Future Forecasting (2025)
- Used population projections and historical weather averages.  
- Forecasts suggest water levels may slightly rise through the end of 2025, though results remain uncertain due to limits of climate predictability.  

---

## Technical Stack
**Languages & Tools:** Python, SQL, Jupyter Notebook, VS Code, GitHub  

**Libraries:**  
- Data Handling → `pandas`, `numpy`, `sqlite3`  
- Visualization → `matplotlib`, `seaborn`  
- Machine Learning → `scikit-learn`, `xgboost`  

---
## Key Results
- Built a robust water level forecasting model for Lake Travis.  
- Demonstrated strong predictive accuracy when lag features are available.  
- Showed feasibility of forward-fill forecasting when future lagged features are missing.  
- Produced interpretable feature importance insights for water policy discussions.  

---

## Challenges & Limitations
- Long-term climate forecasts remain highly uncertain.  
- Missing data in historical water usage required imputation.  
- Future projections rely on averages rather than true climate models.  

---

## Running the Project
  ```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
# Install dependencies
pip install -r requirements.txt
# Open the notebook
jupyter notebook scarcity_cleaning.ipynb
