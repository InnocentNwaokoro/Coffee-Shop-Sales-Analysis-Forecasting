---
# Project Title
---
## ☕ Coffee Shop Sales Analysis & Forecasting
---
## Table of contents




## 📌 Introduction  
Understanding customer behavior and sales trends is crucial for any business, especially in the coffee industry. This project explores sales data from a coffee shop to uncover key insights, optimize decision-making, and build predictive models for future sales trends.  

Using data cleaning, exploratory data analysis (EDA), and time-series forecasting, this analysis aims to help businesses improve **pricing strategies, inventory management, and customer demand prediction**.  

By leveraging **SARIMA (Seasonal ARIMA)** models, we forecast future sales trends, helping businesses stay ahead of demand fluctuations. This project also provides **actionable insights**, such as the best-selling products, peak sales hours, and store performance across different locations.

---

## 🎯 Objectives  
✔️ **Clean and preprocess** sales transaction data (handling missing values, duplicates, outliers, and type conversions).  

✔️ Perform **Exploratory Data Analysis (EDA)** to gain insights into sales patterns, peak coffee consumption times, and store performance.

✔️ Implement **time-series forecasting (SARIMA)** to predict future coffee shop sales trends.  

✔️ Provide **business recommendations** for inventory optimization, staffing efficiency, and targeted promotions.  

---

## 🛠️ Technologies Used  
- **Python (Pandas, NumPy, Matplotlib, Seaborn, Statsmodels)** – Data processing, visualization, and modeling  
- **Time-Series Models (SARIMA, ARIMA)** – Sales forecasting  
- **GitHub** – Version control and documentation  
- **Jupyter Notebooks** – Interactive data analysis and modeling  

---

## 📂 Dataset Source  
This dataset was obtained from **Maven Analytics Data Playground**.  

🔗 [Click here to access the dataset](https://mavenanalytics.io/data-playground?page=6&pageSize=5)  

---

## 📈 Expected Insights  
- Which products are the most popular?** *(Product demand insights)*

![product_demand](https://github.com/user-attachments/assets/fc5cf9fe-61cb-40bd-bdbe-065ac104e7d7)

- What time of the day do people drink the most coffee?** *(Peak coffee consumption hours)*

![coffee_sales_by_hour](https://github.com/user-attachments/assets/44e16b6d-e569-4c77-9f03-b928c970c3bf)

  
- Which days of the week have the highest sales?** *(Sales distribution by day)*

![sales_by_day_of_week](https://github.com/user-attachments/assets/7d02f03e-eebb-4b00-8320-a82ddafc4d4b)
  
- Which month has the highest sales?** *(Monthly sales trends)*

![sales_jan_to_june](https://github.com/user-attachments/assets/0082d8fe-09de-4938-81f6-2391acd7d574)


- How do sales vary by store location?** *(Store performance analysis)*

![store_performance](https://github.com/user-attachments/assets/791e28b9-4ced-4e7e-bf24-ffc2f5709566)


- What is the average sales per transaction?** *(Customer purchasing behavior)*

![transaction_quantity_distribution](https://github.com/user-attachments/assets/52ed128d-89f5-4ffa-93e9-28f5e08635c0)

- What are the overall sales trends over time?** *(Daily/weekly/monthly sales trends)*

![sales_trend](https://github.com/user-attachments/assets/34d48d3f-828e-437c-a6fa-e39ecc446284)

- How can we predict future sales trends?** *(Sales forecasting using SARIMA)*  

![sales_forecasting_sarima](https://github.com/user-attachments/assets/9479edb1-ba51-4fff-a5b4-3f71aec1f285)

---
 
## Data Analysis
### Data Cleaning & Preprocessing 

```python
import pandas as pd
import numpy as np

# Load dataset
file_path = "Coffee Shop Sales.csv"
df = pd.read_csv(file_path)

# Handling missing values
df.dropna(inplace=True)

# Removing duplicates
df.drop_duplicates(inplace=True)

# Convert data types
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S').dt.time
df['transaction_qty'] = df['transaction_qty'].astype(int)
df['unit_price'] = df['unit_price'].astype(float)

# Handling outliers in unit_price using the IQR method
Q1 = df['unit_price'].quantile(0.25)
Q3 = df['unit_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['unit_price'] >= lower_bound) & (df['unit_price'] <= upper_bound)]

# Save cleaned data
df.to_csv("Coffee_Shop_Sales_Cleaned.csv", index=False)

print("Data cleaning completed. Cleaned dataset saved as Coffee_Shop_Sales_Cleaned.csv")
```

### Display summary statistics for numeric columns
```python
df.describe()
```

### Dataset Summary Statistics

Below is the descriptive summary of key numerical features in the dataset:

| Statistic       | transaction_id | transaction_qty | store_id | product_id | unit_price |
|----------------|---------------|----------------|----------|------------|------------|
| **Count**      | 144,904        | 144,904        | 144,904  | 144,904    | 144,904    |
| **Mean**       | 74,720.94      | 1.45           | 5.34     | 48.63      | 3.03       |
| **Std Dev**    | 43,154.33      | 0.54           | 2.07     | 17.04      | 0.82       |
| **Min**        | 1.00           | 1.00           | 3.00     | 22.00      | 0.80       |
| **25% (Q1)**   | 37,253.75      | 1.00           | 3.00     | 34.00      | 2.50       |
| **Median (Q2)**| 74,823.50      | 1.00           | 5.00     | 47.00      | 3.00       |
| **75% (Q3)**   | 112,080.25     | 2.00           | 8.00     | 60.00      | 3.50       |
| **Max**        | 149,456.00     | 4.00           | 8.00     | 87.00      | 4.75       |


---
### Fit SARIMA Model

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# Ensure transaction_date is a datetime index
sales_data.index = pd.to_datetime(sales_data.index)

# Fit SARIMA model (adjust seasonal order based on detected seasonality)
model_sarima = SARIMAX(sales_data, 
                        order=(2, 1, 2),      # (p, d, q) - Non-seasonal parameters
                        seasonal_order=(1, 1, 1, 7),  # (P, D, Q, S) - Seasonal parameters
                        enforce_stationarity=False, 
                        enforce_invertibility=False)

# Train the model
sarima_result = model_sarima.fit()

# Print model summary
print(sarima_result.summary())

# Forecast the next 30 days
forecast_sarima = sarima_result.forecast(steps=30)

# Create forecast dataframe
forecast_dates = pd.date_range(start=sales_data.index[-1] + pd.Timedelta(days=1), periods=30)
forecast_df_sarima = pd.DataFrame({'transaction_date': forecast_dates, 'forecast': forecast_sarima.values})
forecast_df_sarima.set_index('transaction_date', inplace=True)

# Print the forecasted values
print("\n📈 SARIMA Seasonal Forecast (Next 30 Days):\n")
print(forecast_df_sarima)
```

##  SARIMAX Model Results (Seasonal Forecasting)

Below are the results from the **SARIMA(2,1,2)(1,1,[1],7) model**, applied to forecast coffee shop sales.

---

### ** Model Summary**
| Metric                         | Value      |
|---------------------------------|------------|
| **Dependent Variable**          | transaction_qty |
| **Number of Observations**      | 181        |
| **Model Type**                  | SARIMA(2,1,2)(1,1,[1],7) |
| **Log Likelihood**              | -1005.669  |
| **AIC (Akaike Info Criterion)** | 2025.337   |
| **BIC (Bayesian Info Criterion)** | 2046.993  |
| **HQIC**                        | 2034.129   |
| **Sample Period**               | 01-01-2023 to 06-30-2023 |
| **Covariance Type**             | OPG (Outer Product of Gradients) |

---

### **📌 Model Coefficients**
| Parameter | Coefficient | Std. Error | z-score | P-value | Confidence Interval (95%) |
|-----------|------------|------------|---------|---------|---------------------------|
| **AR(1)** | -0.5077    | 0.185      | -2.751  | 0.006   | [-0.869, -0.146] |
| **AR(2)** | 0.0829     | 0.131      | 0.634   | 0.526   | [-0.173, 0.339] |
| **MA(1)** | -0.0506    | 0.165      | -0.307  | 0.759   | [-0.374, 0.273] |
| **MA(2)** | -0.7349    | 0.162      | -4.547  | 0.000   | [-1.052, -0.418] |
| **Seasonal AR(7)** | -0.0543  | 0.121      | -0.449  | 0.654   | [-0.292, 0.183] |
| **Seasonal MA(7)** | -0.8738  | 0.067      | -13.053 | 0.000   | [-1.005, -0.743] |
| **Sigma² (Variance)** | 12850  | 1494.399   | 8.600   | 0.000   | [9922.905, 15800] |

---

### **📌 Diagnostic Tests**
| Test | Value | P-Value |
|------|-------|---------|
| **Ljung-Box (L1) (Q)** | 0.07  | 0.78 (No significant autocorrelation) |
| **Jarque-Bera (JB) Test** | 8.57  | 0.01 (Residuals slightly non-normal) |
| **Heteroskedasticity (H)** | 2.74  | 0.00 (Presence of heteroskedasticity) |
| **Skewness** | -0.51  | - |
| **Kurtosis** | 3.46  | - |

---

### ** SARIMA Seasonal Forecast (Next 30 Days)**
Below is the **30-day forecasted sales** based on the trained **SARIMA seasonal model**.

| **Date**       | **Forecasted Sales** |
|---------------|---------------------|
| 2023-07-01   | 1666.79  |
| 2023-07-02   | 1705.17  |
| 2023-07-03   | 1741.89  |
| 2023-07-04   | 1743.25  |
| 2023-07-05   | 1685.61  |
| 2023-07-06   | 1753.65  |
| 2023-07-07   | 1749.49  |
| 2023-07-08   | 1758.93  |
| 2023-07-09   | 1739.94  |
| 2023-07-10   | 1794.33  |
| 2023-07-11   | 1789.86  |
| 2023-07-12   | 1720.93  |
| 2023-07-13   | 1774.48  |
| 2023-07-14   | 1780.14  |
| 2023-07-15   | 1793.60  |
| 2023-07-16   | 1779.09  |
| 2023-07-17   | 1831.65  |
| 2023-07-18   | 1828.04  |
| 2023-07-19   | 1759.39  |
| 2023-07-20   | 1813.95  |
| 2023-07-21   | 1818.93  |
| 2023-07-22   | 1832.26  |
| 2023-07-23   | 1817.45  |
| 2023-07-24   | 1870.14  |
| 2023-07-25   | 1866.47  |
| 2023-07-26   | 1797.81  |
| 2023-07-27   | 1852.30  |
| 2023-07-28   | 1857.34  |
| 2023-07-29   | 1870.67  |
| 2023-07-30   | 1855.87  |

---

## Interpretation & Key Takeaways

- **AIC & BIC are lower than the previous model**, indicating a better fit.
  
- **Ljung-Box test (p = 0.78)** suggests **no autocorrelation in residuals** → **Good model behavior**.

- **Seasonal MA(7) is highly significant (p < 0.001)** → **The model successfully captures seasonal trends**.
   
- **Heteroskedasticity detected (p = 0.00)** → Variance in residuals changes over time, which **could impact forecast stability**.
  
- **AR(2) & Seasonal AR(7) coefficients are not statistically significant (p > 0.05)** → The model may need **further tuning**.  

---


## 🎯 Conclusion & Recommendation

###  Conclusion  
This project successfully analyzed **coffee shop sales trends** and developed a **seasonal SARIMA model** to forecast future sales.  

Through **Exploratory Data Analysis (EDA)**, we identified key insights such as:  
- Peak coffee consumption hours.  
- Best-performing stores and top-selling products.  
- Seasonal demand variations across days and months.  

The **SARIMA(2,1,2)(1,1,1,7) model** was applied for time-series forecasting, which provided a reliable **30-day sales forecast**.  
This forecast is valuable for **inventory management, staffing, and strategic pricing decisions**.  

---


###  Recommendations  
#### ** Data-Driven Business Strategies**
1 **Optimize Inventory & Supply Chain**  
   - Stock up on high-demand products based on the **forecasted peak sales periods**.  
   - Reduce waste by **adjusting inventory** during low-sales periods.  

2️ **Improve Staffing Efficiency**  
   - Allocate more staff during **peak sales hours** to reduce waiting time and improve service.  
   - Reduce labor costs by **optimizing workforce schedules** based on demand patterns.  

3️ **Dynamic Pricing & Promotions**  
   - Offer discounts on **slow-selling products** during off-peak hours.  
   - Introduce **bundle offers** for frequently bought-together items.  

4️ **Expand High-Performing Store Locations**  
   - Consider opening new stores in **areas with consistently high sales performance**.  
   - Optimize marketing efforts for **underperforming locations**.  

5️ **Enhance Customer Retention Strategies**  
   - Launch a **loyalty program** to encourage repeat purchases.  
   - Offer personalized **discounts based on purchase history**.  

---

## Final Thoughts  
This analysis provides a **data-driven approach to boosting coffee shop revenue** while optimizing **inventory, staffing, and marketing strategies**.  
By continuously improving the forecasting model and incorporating real-time analytics, businesses can **stay ahead of market trends** and maximize profits.  

---

**Next Steps:** Implement these recommendations and track improvements in business performance.  















