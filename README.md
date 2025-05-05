# 🚆 Freight Rail Transport Analysis & Forecasting in Germany (2011–2023) using R

This project is a part of module Application in Data Analytics during my master degree at TU Dresden, Germany. This showcases an end-to-end data science workflow in R — from data cleaning to forecasting transport volume. The dataset used were collacted from various public dataset in [
 - Statistisches Bundesamt]
(https://www.destatis.de/DE/Home/_inhalt.html) 

## 📌 Project Overview
- **Goal:** Exploring transport vollume transport pattern over the period of over 10 years and Forecast transport volume (e.g., freight, passengers) using historical data
- **Language**: R
- **Libraries**: `tidyverse`, `mlr3`, `mlr3tuning`, `xgboost`, `randomForest`, `ggplot2`, `sf`, `tmap`
- **ML Techniques**: Nested resampling, feature selection, hyperparameter tuning

## 📈 Methods
- Data Extract, Transform, and Load (ETL)
- Exploratory Data Analysis (EDA)
- Spatial & Temporal Data Visualization
- Machine Learning model development
- Machine Learning model optimization and Interpretation 

## 📁 1. ETL (Extract, Transform, Load)
### 🔧 Libraries Used
- `readr`, `data.table`, `dplyr` – Data loading & manipulation
- `sf`, `tmap`, `geojsonio` – Spatial data handling

### 🎯 Tasks & Goals
- Load historical freight transport data (2011–2023)
- Merge with GDP and industry production indices
- Normalize inconsistent formats and encode categorical variables
- Filter origin-destination pairs with consistent transport across all 13 years
- Remove missing entries and irrelevant columns

### ✅ Outcomes
- Clean dataset with 11,726 observations and 11 features
- Features include volume, distance, GDP (origin/destination), and sectoral indices
## 📊 2. Exploratory Data Analysis (EDA)

### 🧰 Libraries Used
- `ggplot2`, `dplyr`, `summarytools`

### 🔍 Tasks & Goals
- Describe transport performance and volume distributions
- Detect outliers and assess feature ranges
- Analyze correlations and multicollinearity between variables

### ✅ Outcomes
- Identified key contributing goods (10 types = ~75% of total volume)
- Strong correlation between `rail_volume_ton` and target (`transport_performance`)
- Minimal missing data and no extreme outliers affecting modeling

## 🗺️ 3. Visualization

### 🧰 Libraries Used
- `ggplot2`, `tmap`, `sf`, `gridExtra`

### 🔍 Tasks & Goals
- Visualize temporal trends by goods over years
- Map spatial patterns by origin/destination
- Highlight active transport routes (combined time-space analysis)

### ✅ Outcomes
- Identified temporal drops (e.g., 2018 automotive peak)
- Spatial hotspots: Düsseldorf, Braunschweig, Arnsberg
- Routes like Düsseldorf → Sachsen-Anhalt show 13-year consistency

📸 _Example plots:_

![Goods Trend Over Time](images/goods_trend_over_time.png)
*Fig: Transport performance of top goods over 13 years*

![Transport Heatmap](images/transport_heatmap.png)
*Fig: Year-round active transport routes for coal/lignite*


## 📖 Citation 
Checkout detailed project report for deeper understanding here, and if using this project for academic purposes, please cite the original seminar paper or credit the author via GitHub.
**Thu Thuy Nguyen** - MSc. Transport Economics – TU Dresden, Germany

