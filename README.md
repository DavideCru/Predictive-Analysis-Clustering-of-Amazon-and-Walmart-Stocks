# Predictive-Analysis-Clustering-of-Amazon-and-Walmart-Stocks

Project Type: Machine Learning & Financial Time Series Analysis
Tools Used: Python, yfinance, scikit-learn (Random Forest, K-Means), RMSE, R², Correlation Analysis, Matplotlib

This project analyzes and compares the predictive behavior of two major retail stocks — Amazon (AMZN) and Walmart (WMT) — using machine learning models and clustering techniques.

The objective was twofold:
• Evaluate the predictive performance of a Random Forest regression model on 2023 closing prices.
• Identify hidden structural patterns in historical price dynamics through K-Means clustering.

🔎 Methodology
• Historical daily closing prices collected (2019–2023)
• Feature engineering using lag variables (1–5 days)
• Optimal cluster selection through the Elbow Method (k = 3)
• K-Means clustering for regime segmentation
• Random Forest regression models trained separately for AMZN and WMT
• Model evaluation using RMSE and R²
• Correlation analysis between AMZN and WMT

📊 Key Results
• Amazon (AMZN): 
  - RMSE = 3.77
  - R² = 0.96
→ Excellent predictive performance with strong variance explanation

• Walmart (WMT):
  - RMSE = 2.04
  - R² = 0.42
→ Limited explanatory power despite moderate absolute error

• Moderate positive correlation between AMZN and WMT (ρ = 0.54)
• K-Means clustering (k = 3) revealed distinct and well-separated market regimes
• Confusion matrix confirmed strong separability between clusters

🧠 Conclusion
The project highlights how predictive performance is asset-dependent.
Random Forest proved highly effective for Amazon, capturing 96% of variance in 2023 prices, while performance on Walmart was significantly weaker, suggesting different underlying market dynamics.
Clustering and correlation analysis further confirmed structural differences between the two stocks, reinforcing the importance of adapting modeling strategies to the specific characteristics of each asset.
This project demonstrates financial time-series modeling, supervised regression, unsupervised learning, model evaluation, and critical interpretation of predictive results.

<p align="center">
<img width="45%" alt="image" src="https://github.com/user-attachments/assets/3231169a-f5b9-4aa6-8ad4-bfb83efd7ba1" />
<img width="42%" alt="image" src="https://github.com/user-attachments/assets/21f06e50-79cd-4310-8e7f-7012783b2a2b" />
<img width="50%" alt="image" src="https://github.com/user-attachments/assets/c54adc4c-16dc-478c-8413-3e7a06456c37" />
<img width="50%" alt="image" src="https://github.com/user-attachments/assets/bebde029-fb47-42c4-b3e3-6f7d8bd4e06c" />
</p>
 


