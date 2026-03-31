# Municipal Waste Demand Forecasting

## Project Overview

Waste management operations depend heavily on predicting how much waste will be generated in the future. If forecasts are inaccurate, cities either allocate too many resources (wasting money) or too few (causing service delays and operational headaches).

In this project, I built a **time-series forecasting model** that predicts monthly municipal waste demand using historical waste collection data. The goal was to understand patterns in waste generation and create a model that could help operations teams plan resources more effectively.

This project walks through a full forecasting workflow including:

- Data cleaning
- Exploratory data analysis
- Seasonality detection
- Feature engineering
- Machine learning forecasting
- Model evaluation
- Forecast visualization

---

## Dataset

The dataset contains **monthly records of different types of waste collected** over approximately 11 years.

Waste categories include:

- Residential waste
- Public litter bins
- Dumped rubbish
- Street sweepings
- Mattresses
- Commingled recycling
- Cardboard
- Hard waste sent to landfill
- Hard waste recovered
- Green waste

Each row represents a **single month of waste collection data**.

To simplify forecasting, a new variable **`total_waste`** was created by summing all waste categories.

---

## Data Cleaning

Before modeling, the dataset was cleaned and structured:

- Removed redundant columns such as `month` and `hardwaste_total`
- Converted the date column into a proper **datetime index**
- Sorted the dataset chronologically
- Checked for missing values
- Ensured a consistent **monthly time frequency**

After these steps the dataset became a clean time-series ready for analysis.

---

## Exploratory Data Analysis

Several visualizations were created to understand the dataset.

### Trend Analysis
The total waste collected showed a **steady upward trend** over time, likely driven by population growth and increased consumption.

### Seasonality
Time series decomposition revealed **clear yearly seasonal patterns** in waste generation.

### Monthly Distribution
A monthly boxplot showed that certain months consistently generate more waste than others, confirming the presence of seasonal demand cycles.

---

## Feature Engineering

Because this is time-series data, past values provide useful signals for predicting future demand.

### Lag Features

- `waste_last_month`
- `waste_3_months_ago`
- `waste_6_months_ago`
- `waste_last_year`

These allow the model to learn how past waste levels influence future values.

### Rolling Statistics

- `rolling_mean_3`
- `rolling_mean_6`
- `rolling_std_6`

These features capture **short-term trends and volatility** in the waste collection system.

---

## Model

A **Random Forest Regressor** was used to forecast total waste demand.

Reasons for choosing Random Forest:

- Handles nonlinear relationships well
- Works effectively on tabular datasets
- Robust to noise
- Requires minimal feature scaling

The model was trained using a **time-based train/test split** to preserve chronological order and simulate real forecasting conditions.

---

## Model Evaluation

Two metrics were used to evaluate forecasting accuracy:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

Results:

MAE ≈ **219**  
RMSE ≈ **266**

Given that waste volumes range roughly between **3000 and 5200**, the model achieves an average prediction error of about **4–5%**, which is reasonable for operational forecasting.

---

## Feature Importance

Feature importance analysis showed that **rolling averages of recent waste levels** were the strongest predictors.

This suggests that **recent trends in waste generation are highly informative for forecasting future demand**.

---

## Results

The model successfully captures the **overall trend and seasonal behavior** of municipal waste demand.

Although the model smooths some short-term spikes, it follows the general pattern of the real data quite closely.

Final prediction comparison:

![Forecast Results](waste_forecast_results.png)

---

## Project Pipeline

Data Cleaning
↓
Exploratory Data Analysis
↓
Seasonality Detection
↓
Feature Engineering
↓
Random Forest Forecasting
↓
Model Evaluation
↓
Forecast Visualization


---

## Future Improvements

Possible improvements include:

- Comparing Random Forest with classical models such as **SARIMA** or **Prophet**
- Implementing **walk-forward validation**
- Incorporating additional predictors such as weather or population data
- Building a dashboard for operational decision support

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

---

## Author

Built as a time-series forecasting project exploring how machine learning can be applied to operational demand prediction in waste management systems.

